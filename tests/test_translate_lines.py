"""翻译阶段的编排与 id 对齐校验。

id 对齐是这个文件里最该锁死的东西：模型漏一条、合并两条，中文就跟时间戳整条错位，
而那种错只有看成片字幕时才发现，且是从错位那句往后全错。
"""

import json

import pytest

from tenmin.config import ProjectConfig
from tenmin.models import DialogueLine, DialogueTrack, TranslatedLine, TranslatedTrack
from tenmin.script.llm import LLMResponseFormatError, LLMSchemaError
from tenmin.translate import lines as tl
from tenmin.translate.srt_writer import render_zh_srt


def _line(idx: int, text: str, kind: str = "dialogue", start: float = 0.0):
    return DialogueLine(
        idx=idx, start=start, end=start + 1.0, text=text, raw=text, kind=kind
    )


def _track(*items: DialogueLine) -> DialogueTrack:
    return DialogueTrack(episode=11, source="asr", duration=100.0, lines=list(items))


# ---- 选哪些行去翻译 ----


def test_non_speech_lines_are_not_translated():
    """片头片尾的 staff 名单、整行括注不是说出口的话，翻它们是白花钱。"""
    track = _track(
        _line(1, "作曲：某人", kind="credits"),
        _line(2, "本編のセリフ"),
        _line(3, "（電話の音）", kind="screen_text"),
        _line(4, "♪～", kind="noise"),
    )
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [2]


def test_monologue_lines_are_translated():
    """内心独白是真的说出口过的台词（双轨字幕拆出来的那一段），漏掉就少一条中文字幕。"""
    track = _track(_line(1, "セリフ"), _line(1, "モノローグ", kind="monologue"))
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [1, 2]


def test_positions_are_one_based_indexes_into_the_track():
    """id 用位置下标而不是 DialogueLine.idx：后者在双轨字幕被拆开时会重复。"""
    track = _track(_line(7, "あ"), _line(7, "い"))
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [1, 2]


def test_a_track_with_no_dialogue_selects_nothing():
    assert tl.select_translatable(_track(_line(1, "x", kind="credits"))) == []


# ---- prompt 里的对白块 ----


def test_lines_block_is_id_pipe_text():
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    block = tl.build_lines_block(tl.select_translatable(track))
    assert block == "1 | はい\n2 | いいえ"


def test_lines_block_keeps_the_selected_ids_not_a_renumbering():
    track = _track(_line(1, "x", kind="credits"), _line(2, "はい"))
    block = tl.build_lines_block(tl.select_translatable(track))
    assert block == "2 | はい"


# ---- id 对齐校验（核心）----


def test_alignment_accepts_an_exact_match():
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="一"), TranslatedLine(id=2, zh="二")]
    )
    tl.check_alignment({1, 2}, translated)  # 不抛


def test_alignment_rejects_a_missing_id():
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="一")])
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1, 137, 298}, translated)
    message = str(excinfo.value)
    assert "137" in message and "298" in message


def test_alignment_rejects_an_unknown_id():
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="一"), TranslatedLine(id=999, zh="?")]
    )
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1}, translated)
    assert "999" in str(excinfo.value)


def test_alignment_rejects_a_duplicated_id():
    """条数对上了但有重复 —— 单看条数查不出来，必须按集合判。"""
    translated = TranslatedTrack(
        episode=11,
        lines=[
            TranslatedLine(id=1, zh="一"),
            TranslatedLine(id=1, zh="壹"),
            TranslatedLine(id=3, zh="三"),
        ],
    )
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1, 2, 3}, translated)
    message = str(excinfo.value)
    assert "多次" in message and "1" in message


def test_alignment_error_message_is_bounded():
    """回灌给模型的消息不能是三百个数字 —— 那会挤掉真正要修的内容。"""
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment(set(range(1, 401)), TranslatedTrack(episode=11, lines=[]))
    assert len(str(excinfo.value)) < 600


def test_alignment_error_survives_the_repair_layer_truncation():
    """消息要完整进回灌，就得短于 schema 修复层那个截断上限。"""
    from tenmin.script.llm import REPAIR_ERROR_MAX_CHARS

    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment(set(range(1, 401)), TranslatedTrack(episode=11, lines=[]))
    assert len(str(excinfo.value)) < REPAIR_ERROR_MAX_CHARS


# ---- 编排 ----


class _ScriptedProvider:
    """按脚本逐次返回预设的响应，记下每次收到的 prompt。

    签名跟 LLMProvider.complete 一致（system 与 user 两个位置参数）。
    """

    def __init__(self, *payloads: str):
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.schemas: list[object] = []

    async def complete(self, system, user, schema=None):
        self.systems.append(system)
        self.prompts.append(user)
        self.schemas.append(schema)
        return self.payloads[len(self.prompts) - 1]


def _payload(ids, glossary=None, episode: int = 11) -> str:
    return json.dumps(
        {
            "episode": episode,
            "lines": [{"id": i, "zh": f"译{i}"} for i in ids],
            "glossary": glossary or {},
        },
        ensure_ascii=False,
    )


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(show="测试番", slug="test").bind_root(tmp_path)


@pytest.mark.asyncio
async def test_translate_track_returns_the_aligned_result(tmp_path):
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1, 2], {"リディア": "莉迪亚"}))

    result = await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    assert [line.id for line in result.lines] == [1, 2]
    assert result.glossary == {"リディア": "莉迪亚"}
    assert result.episode == 11


@pytest.mark.asyncio
async def test_translate_track_asks_for_text_not_a_schema_object(tmp_path):
    """schema 已经写进 prompt，校验与重试由这一层自己接管。"""
    provider = _ScriptedProvider(_payload([1]))
    await tl.translate_track(
        _cfg(tmp_path), _track(_line(1, "はい")), provider, accumulated={}
    )
    assert provider.schemas == [None]


@pytest.mark.asyncio
async def test_the_prompt_carries_the_schema_and_the_lines(tmp_path):
    provider = _ScriptedProvider(_payload([1]))
    await tl.translate_track(
        _cfg(tmp_path), _track(_line(1, "はい")), provider, accumulated={}
    )
    prompt = provider.prompts[0]
    assert "1 | はい" in prompt
    assert "TranslatedLine" in prompt  # schema 正文
    assert provider.systems[0].strip()  # 系统提示不是空串


@pytest.mark.asyncio
async def test_the_episode_comes_from_the_track_not_the_model(tmp_path):
    """集号是我们自己的事实，模型报错了不能让它污染产物文件名与下游。"""
    track = _track(_line(1, "はい"))
    provider = _ScriptedProvider(_payload([1], episode=99))

    result = await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    assert result.episode == 11


@pytest.mark.asyncio
async def test_translate_track_retries_a_misaligned_response(tmp_path):
    """第一轮漏了第 2 条，回灌重试后补齐。"""
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1]), _payload([1, 2]))

    result = await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    assert [line.id for line in result.lines] == [1, 2]
    assert len(provider.prompts) == 2


@pytest.mark.asyncio
async def test_the_retry_prompt_still_carries_the_source_lines(tmp_path):
    """要补的是「漏掉那几条的译文」，模型得对着原文才补得出来。"""
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1]), _payload([1, 2]))

    await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    retry = provider.prompts[1]
    assert "2 | いいえ" in retry
    assert "逐条对齐" in retry  # 上一轮的报错被回灌了


@pytest.mark.asyncio
async def test_translate_track_gives_up_after_the_retry_budget(tmp_path):
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(*[_payload([1])] * 10)

    with pytest.raises(LLMSchemaError):
        await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    assert len(provider.prompts) == _cfg(tmp_path).llm.max_attempts


@pytest.mark.asyncio
async def test_the_prompt_carries_the_accumulated_glossary(tmp_path):
    track = _track(_line(1, "はい"))
    provider = _ScriptedProvider(_payload([1]))

    await tl.translate_track(
        _cfg(tmp_path), track, provider, accumulated={"リディア": "莉迪亚"}
    )

    assert "莉迪亚" in provider.prompts[0]


@pytest.mark.asyncio
async def test_manual_glossary_overrides_the_accumulated_one_in_the_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.glossary = {"リディア": "莉蒂亚"}
    provider = _ScriptedProvider(_payload([1]))

    await tl.translate_track(
        cfg, _track(_line(1, "はい")), provider, accumulated={"リディア": "莉迪亚"}
    )

    assert "莉蒂亚" in provider.prompts[0]
    assert "莉迪亚" not in provider.prompts[0]


@pytest.mark.asyncio
async def test_a_track_without_dialogue_short_circuits(tmp_path):
    """一条对白都没有就别发请求。"""
    provider = _ScriptedProvider()
    result = await tl.translate_track(
        _cfg(tmp_path), _track(_line(1, "x", kind="credits")), provider, accumulated={}
    )
    assert result.lines == []
    assert result.episode == 11
    assert provider.prompts == []


# ---- 跟字幕交付物对接（id 的语义由下游锁住）----


@pytest.mark.asyncio
async def test_ids_still_map_back_to_the_original_lines_after_filtering(tmp_path):
    """过滤掉若干行之后，id 必须仍是**原轨**里的位置，而不是送进模型那份清单的下标。

    render_zh_srt 直接做 `track.lines[line.id - 1]`，所以这两种编号法差一位就会让整份
    中文字幕系统性错位。
    """
    track = _track(
        _line(1, "staff", kind="credits", start=0.0),
        _line(2, "はい", start=10.0),
        _line(3, "（音）", kind="screen_text", start=20.0),
        _line(4, "いいえ", start=30.0),
    )
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [2, 4]

    provider = _ScriptedProvider(_payload([2, 4]))
    result = await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    srt = render_zh_srt(track, result)
    assert "00:00:10,000 --> 00:00:11,000\n译2" in srt
    assert "00:00:30,000 --> 00:00:31,000\n译4" in srt
