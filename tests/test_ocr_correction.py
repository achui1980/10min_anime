"""OCR 整集校对：只采用合格修改、保留不确定文本、缓存与人工修改保护。"""

import asyncio
import json
from pathlib import Path

import pytest

from tenmin.config import LLMConfig
from tenmin.ingest.asr import render_srt
from tenmin.ingest.ocr_correction import apply_fixes, correct_cues, select_corrected
from tenmin.ingest.srt_parser import load_srt
from tenmin.models import RawCue


def _cues():
    return [
        RawCue(idx=1, start=1, end=2, text="垃圾也杀丢"),
        RawCue(idx=2, start=3, end=4, text="汇兕"),
    ]


class Provider:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []
        self.closed = False

    async def complete(self, system, user, schema):
        self.calls.append(user)
        if self.error:
            raise self.error
        return schema.model_validate({"lines": self.results})

    async def aclose(self):
        self.closed = True


def _results():
    return [
        {"id": 1, "original": "垃圾也杀丢", "text": "垃圾也没丢", "reason": "形近字"},
        {
            "id": 2, "original": "汇兕", "text": "汇兑",
            "uncertain": True, "reason": "仅凭文字无法确认",
        },
    ]


def _fix(id, original, text, **extra):
    from tenmin.ingest.ocr_correction import _Fix

    return _Fix(id=id, original=original, text=text, **extra)


def test_confident_change_keeps_timing_and_uncertain_line_unchanged():
    provider = Provider(_results())
    corrected, review, dropped = asyncio.run(correct_cues(_cues(), provider))
    assert [(c.idx, c.start, c.end, c.text) for c in corrected] == [
        (1, 1, 2, "垃圾也没丢"),
        (2, 3, 4, "汇兕"),
    ]
    assert [(r.id, r.text, r.reason, r.candidate) for r in review] == [
        (2, "汇兕", "仅凭文字无法确认", "汇兑")
    ]
    assert not dropped
    assert len(provider.calls) == 1
    assert "1 | 1.00-2.00 | 垃圾也杀丢" in provider.calls[0]
    assert "日语" not in provider.calls[0]


def test_unreturned_lines_stay_unchanged():
    corrected, review, dropped = asyncio.run(correct_cues(_cues(), Provider([])))
    assert corrected == _cues() and not review and not dropped


@pytest.mark.parametrize(
    ("fix", "why"),
    [
        (("3", "x", "y"), "越界"),
        ((1, "垃圾也杀丢", "   "), "为空"),
        ((1, "垃圾也杀丢", "台词 00:00:05,000 --> 00:00:06,000"), "时间戳"),
        ((1, "完全不同的另一句话", "垃圾也没丢"), "错位"),
    ],
)
def test_invalid_fixes_are_dropped(fix, why):
    id, original, text = fix
    cues, review, dropped = apply_fixes(_cues(), [_fix(int(id), original, text)])
    assert cues == _cues() and not review
    assert len(dropped) == 1 and why in dropped[0]


def test_duplicate_id_keeps_first_and_drops_rest():
    cues, _review, dropped = apply_fixes(_cues(), [
        _fix(1, "垃圾也杀丢", "垃圾也没丢"),
        _fix(1, "垃圾也杀丢", "垃圾也杀了"),
    ])
    assert cues[0].text == "垃圾也没丢"
    assert len(dropped) == 1 and "重复" in dropped[0]


def test_large_rewrite_goes_to_review_without_applying():
    cues = [RawCue(idx=1, start=0, end=1, text="经常被称为三大功臣")]
    corrected, review, dropped = apply_fixes(
        cues, [_fix(1, "经常被称为三大功臣", "被统称为建国的三大英雄")]
    )
    assert corrected == cues and not dropped
    assert review[0].candidate == "被统称为建国的三大英雄"
    assert "改动过大" in review[0].reason


def test_asr_text_is_included_in_prompt(tmp_path):
    provider = Provider([])
    asr = [RawCue(idx=1, start=1, end=2, text="じんし様")]
    asyncio.run(correct_cues(_cues(), provider, asr=asr))
    assert "日语语音转写" in provider.calls[0]
    assert "1.00-2.00 | じんし様" in provider.calls[0]


def _paths(tmp_path: Path):
    raw = tmp_path / "E01.ocr.srt"
    raw.write_text(render_srt(_cues()), encoding="utf-8")
    return (
        raw, tmp_path / "E01.ocr.corrected.srt", tmp_path / "state.json", tmp_path / "review.json"
    )


def _select(paths, factory, **extra):
    raw, corrected, state, review = paths
    return asyncio.run(
        select_corrected(
            raw,
            corrected,
            state,
            review,
            provider_factory=factory,
            cfg=LLMConfig(),
            source_identity="source-v1",
            **extra,
        )
    )


def test_success_caches_and_preserves_human_edit(tmp_path):
    paths = _paths(tmp_path)
    provider = Provider(_results())
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert (chosen, warnings, retry) == (paths[1], [], False)
    assert provider.closed
    assert [c.text for c in load_srt(paths[1])] == ["垃圾也没丢", "汇兕"]
    assert json.loads(paths[3].read_text())[0]["reason"] == "仅凭文字无法确认"
    assert json.loads(paths[2].read_text())["status"] == "success"
    assert _select(paths, lambda: pytest.fail("must reuse"))[0] == paths[1]
    paths[1].write_text(render_srt([c.model_copy(update={"text": "手工"}) for c in _cues()]))
    assert _select(paths, lambda: pytest.fail("must preserve human edit"))[0] == paths[1]
    assert load_srt(paths[1])[0].text == "手工"


def test_dropped_fixes_are_reported_as_warning(tmp_path):
    paths = _paths(tmp_path)
    results = [*_results(), {"id": 9, "original": "x", "text": "y"}]
    chosen, warnings, retry = _select(paths, lambda: Provider(results))
    assert chosen == paths[1] and not retry
    assert len(warnings) == 1 and "丢弃了 1 条" in warnings[0]


def test_asr_change_invalidates_cache(tmp_path):
    paths = _paths(tmp_path)
    asr = tmp_path / "ja.asr.srt"
    asr.write_text(render_srt([RawCue(idx=1, start=1, end=2, text="ゴミ")]), encoding="utf-8")
    provider = Provider(_results())
    _select(paths, lambda: provider, asr=asr)
    assert "ゴミ" in provider.calls[0]
    assert _select(paths, lambda: pytest.fail("reuse"), asr=asr)[0] == paths[1]
    again = Provider(_results())
    _select(paths, lambda: again)
    assert again.calls


def test_changed_raw_with_human_edit_preserves_old_and_uses_new_raw(tmp_path):
    paths = _paths(tmp_path)
    _select(paths, lambda: Provider(_results()))
    paths[1].write_text("手工终稿")
    paths[0].write_text(render_srt([c.model_copy(update={"text": "新原稿"}) for c in _cues()]))
    chosen, warnings, retry = _select(paths, lambda: pytest.fail("do not overwrite"))
    assert chosen == paths[0] and warnings and retry
    assert paths[1].read_text() == "手工终稿"


def test_missing_key_or_request_failure_falls_back_and_retries(tmp_path):
    paths = _paths(tmp_path)
    calls = 0

    def missing():
        nonlocal calls
        calls += 1
        raise RuntimeError("缺少 key")

    assert _select(paths, missing) == (paths[0], ["OCR AI 校对失败：缺少 key"], True)
    assert _select(paths, missing)[2] and calls == 2
    provider = Provider(error=RuntimeError("请求超时"))
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[0] and retry and "请求超时" in warnings[0]
    assert provider.closed
    assert not paths[2].exists()


def test_unmanaged_existing_corrected_file_is_never_overwritten(tmp_path):
    paths = _paths(tmp_path)
    paths[1].write_text("既有字幕")
    chosen, warnings, retry = _select(paths, lambda: pytest.fail("no provider"))
    assert chosen == paths[0] and warnings and retry
    assert paths[1].read_text() == "既有字幕"


def test_model_change_preserves_human_edit_when_source_is_same(tmp_path):
    paths = _paths(tmp_path)
    _select(paths, lambda: Provider(_results()))
    paths[1].write_text("人工稿")
    chosen, _warnings, retry = asyncio.run(select_corrected(
        *paths, provider_factory=lambda: pytest.fail("must preserve"),
        cfg=LLMConfig(model="other-model"), source_identity="source-v1",
    ))
    assert chosen == paths[1] and not retry


def test_provider_cleanup_failure_warns_and_falls_back(tmp_path):
    paths = _paths(tmp_path)

    class CloseFailure(Provider):
        async def aclose(self):
            raise RuntimeError("关闭连接失败")

    chosen, warnings, retry = _select(paths, lambda: CloseFailure(_results()))
    assert chosen == paths[0] and retry
    assert "关闭连接失败" in warnings[0]
    assert not paths[1].exists()


def test_malformed_raw_is_not_silently_reduced(tmp_path):
    paths = _paths(tmp_path)
    paths[0].write_text(paths[0].read_text() + "\n3\ninvalid timestamp\n不能丢的原文\n")
    provider = Provider(_results())
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[0] and warnings and retry
    assert not provider.calls and not paths[1].exists()


def test_glossary_goes_into_prompt_and_cache_identity(tmp_path):
    paths = _paths(tmp_path)
    provider = Provider(_results())
    _select(paths, lambda: provider, glossary={"壬氏": "壬氏", "猫猫": "猫猫"})
    assert "人物名表" in provider.calls[0]
    assert "猫猫 → 猫猫" in provider.calls[0]
    assert _select(
        paths, lambda: pytest.fail("reuse"), glossary={"猫猫": "猫猫", "壬氏": "壬氏"}
    )[0] == paths[1]
    again = Provider(_results())
    _select(paths, lambda: again, glossary={"猫猫": "猫猫"})
    assert again.calls


def test_no_glossary_keeps_prompt_without_section(tmp_path):
    paths = _paths(tmp_path)
    provider = Provider(_results())
    _select(paths, lambda: provider)
    assert "人物名表" not in provider.calls[0]
