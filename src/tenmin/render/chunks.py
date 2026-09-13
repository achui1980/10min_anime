"""旁白切句与 hold 定位。全是纯函数。

每个 chunk 独立 TTS：真实时长直接测得，不依赖 Edge-TTS 的 word boundary；
单个 chunk 失败只需补它一个。
"""

from __future__ import annotations

import unicodedata

from tenmin.models import Beat, Hold
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

# 一句 = 若干非句末字符 + 一串句末标点；末尾没标点的残句单独成句。
_SENTENCE_ENDINGS = "。！？!?"

# 句末标点之后还该跟着一起走的**收尾**符号。分成两族，因为判据不一样。
#
# 无歧义族：这些字符只可能收尾，见一个就吸收一个，不用看上下文。
_UNAMBIGUOUS_CLOSERS = "）)」』】〉》”’"
# 歧义族：ASCII 的 `'` 与 `"` 同一个字符既当开引号又当收引号（实测 work/saijo 的旁白
# 全用 ASCII `'`，10 集 42 处）。所以只有**出现次数为偶数**的那一个才是收引号；奇数那个
# 是开引号，必须留给它引的那句话。
_AMBIGUOUS_QUOTES = "'\""

# 刻意**不**把 `…` 与 `；` 列进 _SENTENCE_ENDINGS。实测（10 集真实 script.json）：
# 加 `…` 会改掉 1 个 beat 的 chunk 划分、加 `；` 也是 1 个、两个都加是 2 个 —— 而 chunk
# 文本一变，文件名里的内容哈希就变，已经合成好的 mp3 全要重跑、成片音频跟着变。这属于
# 「要不要把 chunk 切细」那个产品决定（见 A5 的调研），不是本次要修的 bug。


def is_pronounceable(text: str) -> bool:
    """这段文本里有没有任何「读得出声」的字符。

    判据用 unicodedata 的大类：L*（字母，含 CJK 汉字与假名）与 N*（数字）算可发音，
    标点（P*）、符号（S*）、空白（Z*）、破折号一概不算。刻意不用字符黑名单：黑名单永远
    补不全，而「没有任何字母或数字」正好是「没有内容可读」的等价表述。

    住在这里而不是 render/tts.py（它原来的家）：split_sentences 的兜底那一层要用同一个
    判据，而 tts.py 已经 import 本模块，反向 import 会成环。tts.py 原样 re-export 它。
    """
    return any(unicodedata.category(char)[0] in "LN" for char in text)


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，标点与紧跟其后的收尾符号都跟着前一句。连续标点算同一句末。

    「收尾符号也跟着走」是 P2-E A1 修的 bug。原来 `。` 之后一律断开，于是
    `…她应尽的义务。'` 里那个收引号自成一句 —— 它不含任何可发音字符，
    render/tts.py 得专门跳过它（否则 Edge TTS 抛 NoAudioReceived），而字幕那边
    直接多出一条 0.18 秒、正文只有一个 `'` 的 cue（生产实证：work/saijo E05 有两条）。

    两层判据，缺一不可：

    1. **奇偶**决定歧义引号的归属。旁白用的是 ASCII `'`，同一个字符两头都用；朴素的
       「终止符后一律吸收」会把开引号粘到上一句尾巴上，引号跟它引的话被拆开。实测这条
       差别不只是观感：它会改掉 E05 两个 chunk 的文本，让已经合成好的 mp3 失效。
       奇偶版在 10 集真实数据上 **chunk 划分零变化**（音频产物全部继续有效）。
    2. **兜底合并**：引号数量不成对时奇偶必然判错（那一刻之后全反），所以最后再扫一遍，
       把「没有任何可发音字符」的句子折进前一句（首句没有前一句，折进下一句）。这条保证
       「切出来的每一句都有内容可读」是**无条件**成立的不变量，而不是「引号写规范时才成立」。

    切句是纯重新分组：拼回去与 strip 过的原文逐字节相同（有测试锁着）。
    """
    sentences: list[str] = []
    buffer: list[str] = []
    pending_end = False
    # 每种歧义引号各数一个计数器：偶数次出现的那个才是收引号。
    quote_counts: dict[str, int] = {}
    for char in text.strip():
        if char == "\n":
            continue
        if char in _AMBIGUOUS_QUOTES:
            quote_counts[char] = quote_counts.get(char, 0) + 1
        if char in _SENTENCE_ENDINGS:
            buffer.append(char)
            pending_end = True
            continue
        if pending_end:
            if char in _UNAMBIGUOUS_CLOSERS or (
                char in _AMBIGUOUS_QUOTES and quote_counts[char] % 2 == 0
            ):
                buffer.append(char)
                continue
            sentences.append("".join(buffer))
            buffer = []
            pending_end = False
        buffer.append(char)
    if buffer:
        sentences.append("".join(buffer))
    stripped = [s for s in (item.strip() for item in sentences) if s]
    return _merge_unpronounceable(stripped)


def _merge_unpronounceable(sentences: list[str]) -> list[str]:
    """把没有任何可发音字符的句子折进邻居，不丢字。

    往**前**折是首选（`。'` 里那个 `'` 属于前面那句话）；只有碎片正好在最前面、
    没有前一句可挂时才往后折。
    """
    merged: list[str] = []
    for sentence in sentences:
        if merged and not is_pronounceable(sentence):
            merged[-1] += sentence
        else:
            merged.append(sentence)
    if len(merged) > 1 and not is_pronounceable(merged[0]):
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged



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
