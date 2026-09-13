"""旁白切句与 hold 定位。全是纯函数。

每个 chunk 独立 TTS：真实时长直接测得，不依赖 Edge-TTS 的 word boundary；
单个 chunk 失败只需补它一个。
"""

from __future__ import annotations

from tenmin.models import Beat, Hold
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

# 一句 = 若干非句末字符 + 一串句末标点；末尾没标点的残句单独成句。
_SENTENCE_ENDINGS = "。！？!?"


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，标点跟着前一句。连续标点算同一句末。"""
    sentences: list[str] = []
    buffer: list[str] = []
    pending_end = False
    for char in text.strip():
        if char == "\n":
            continue
        if char in _SENTENCE_ENDINGS:
            buffer.append(char)
            pending_end = True
            continue
        if pending_end:
            sentences.append("".join(buffer))
            buffer = []
            pending_end = False
        buffer.append(char)
    if buffer:
        sentences.append("".join(buffer))
    return [s for s in (item.strip() for item in sentences) if s]


def sentence_offsets(sentences: list[str]) -> list[float]:
    """每句「结束时刻」的估算值（秒），用 v1 的 4.5 字/秒。

    已知不一致（范围外，未修）：这里**不看 render.rate**，而 script/budget.py 与
    render/tts.py 的时长体检都已经跟着 rate 走了。后果是 rate != "+0%" 时 hold 会被
    assign_holds 按到偏移不对的句边界上（rate=+20% 时偏移应该是这里算出来的 1/1.2）。
    修法就是把 rate 一路传进 plan_chunks，但那要动 plan_chunks / _plan_pronounceable /
    synthesize_track 三层签名，属于 render 侧的改动。
    """
    offsets: list[float] = []
    cursor = 0.0
    for sentence in sentences:
        cursor += narration_chars(sentence) / SPEECH_RATE_CPS
        offsets.append(cursor)
    return offsets


def assign_holds(sentences: list[str], holds: list[Hold]) -> dict[int, float]:
    """把每个 hold 落到最近的句边界上。返回 {句索引: 该句之后的静音秒数}。"""
    if not sentences:
        return {}
    offsets = sentence_offsets(sentences)
    assigned: dict[int, float] = {}
    for hold in holds:
        # 平手取靠前的边界：min 遇到相等的 key 保留第一个
        index = min(range(len(offsets)), key=lambda i: abs(offsets[i] - hold.at))
        assigned[index] = assigned.get(index, 0.0) + hold.duration
    return assigned


def plan_chunks(beat: Beat) -> list[tuple[str, float]]:
    """把一个 beat 切成 [(要合成的文本, 该 chunk 之后的静音秒数)]。

    空旁白返回空列表。这条路**只可能**是人工编辑走出来的：LLM 输出侧
    LLMBeat.narration 是 NonBlankStr，空的进不来。调用方（render/tts.py 的
    _plan_pronounceable）负责把它降级成一条 warning。
    """
    sentences = split_sentences(beat.narration)
    if not sentences:
        return []
    hold_after = assign_holds(sentences, beat.audio.holds)
    chunks: list[tuple[str, float]] = []
    buffer: list[str] = []
    for index, sentence in enumerate(sentences):
        buffer.append(sentence)
        silence = hold_after.get(index)
        if silence is not None:
            chunks.append(("".join(buffer), silence))
            buffer = []
    if buffer:
        chunks.append(("".join(buffer), 0.0))
    return chunks
