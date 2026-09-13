"""旁白切句与 hold 定位。全是纯函数。

每个 chunk 独立 TTS：真实时长直接测得，不依赖 Edge-TTS 的 word boundary；
单个 chunk 失败只需补它一个。
"""

from __future__ import annotations

import re
import unicodedata

from tenmin.models import Beat, Hold
from tenmin.script.budget import DEFAULT_RATE, narration_seconds, narration_span_seconds

# 一句 = 若干非句末字符 + 一串句末标点；末尾没标点的残句单独成句。
_SENTENCE_ENDINGS = "。！？!?"

# 连续的换行（含 `\r\n` 与裸 `\r`）折成**一个空格**。
#
# 原来是见 `\n` 就 `continue`，一个分隔符都不留（而裸 `\r` 压根没被处理，会原样留在
# chunk 文本里、进 TTS 请求与内容哈希）。CJK 不靠空格分词所以看不出来，旁白里嵌拉丁文
# 时就粘成一个词：`"hello\nworld"` → `"helloworld"`，Edge TTS 当一个生词读，字幕上也
# 少一个词界。
#
# 为什么换成空格、而不是当句子边界：换行没有句读语义（LLM 也可能只是在排版），凭它
# 造一个句子会切出没有终止标点的残句、还可能让 hold 落到一个作者没打算断开的位置。
# 空格是最小介入的选择，而且**不改字数**：budget.narration_chars 把全部空白 sub 掉，
# 所以时长估算、hold 定位、字幕分配全都逐点不变。
#
# 实测 11 份真实 script.json 的 73 段 narration：`\n` 与 `\r` 各 0 次，所以这是纯
# 防御性修复，现有产物一个字节都不动。
#
# 顺带吃掉换行**紧邻**的空白，免得 `"甲。\n  乙。"` 留下三个空格；不碰独立出现的空格
# （那可能是作者刻意写的，而语料里压根没有空格字符）。
_LINE_BREAKS = re.compile(r"\s*[\r\n]+\s*")

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

    「收尾符号也跟着走」修的是一个真实 bug。原来 `。` 之后一律断开，于是
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

    切句是纯重新分组：拼回去与 strip 过的原文逐字节相同（有测试锁着）。唯一的例外是
    换行 —— 它会先被折成一个空格（见 _LINE_BREAKS），真实旁白里 0 次出现。
    """
    sentences: list[str] = []
    buffer: list[str] = []
    pending_end = False
    # 每种歧义引号各数一个计数器：偶数次出现的那个才是收引号。
    quote_counts: dict[str, int] = {}
    for char in _LINE_BREAKS.sub(" ", text).strip():
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



def sentence_offsets(sentences: list[str], *, rate: str = DEFAULT_RATE) -> list[float]:
    """每句「结束时刻」的估算值（秒）。

    换算走 budget.narration_seconds，所以它跟 render.rate 是**同一份**语速：
    script/budget.py 的预算与 render/tts.py 的合成结果时长体检都按 rate 缩放，
    而这里曾经写死 4.5 字/秒。后果不是「估算不准」而是**定位错**：assign_holds 把 hold
    贴到「偏移最近的句边界」，rate="+20%" 时真实的句边界比这里算的早 1/1.2，于是留白被
    插到了另一句后面。实测 10 集 52 个带 hold 的 beat：rate="+20%" 会让 29 个的落点变、
    "-20%" 会让 31 个变 —— 也就是说这个缺陷一旦用户动了 rate 就立刻显形。

    `rate` 默认 `"+0%"`（speed_factor 恒为 1.0），所以默认路径与改动前逐点等价。
    """
    offsets: list[float] = []
    cursor = 0.0
    for sentence in sentences:
        cursor += narration_seconds(sentence, rate=rate)
        offsets.append(cursor)
    return offsets


def assign_holds(
    sentences: list[str],
    holds: list[Hold],
    *,
    rate: str = DEFAULT_RATE,
    warnings: list[str] | None = None,
) -> dict[int, float]:
    """把每个 hold 落到最近的句边界上。返回 {句索引: 该句之后的静音秒数}。

    两种坏输入原来是**静默**吞掉的，现在各报一条 warning（传法跟 render/tts.py 的
    `_plan_pronounceable` 一致：调用方给一个列表，这里往里 append）。两条都不改
    返回值 —— 它们不失同步，只是「有个 hold 的落点没实现」，属于本项目惯用的降级：

    1. **两个 hold 落到同一句**：静音时长相加是对的（成片里就是一段更长的静音），
       但其中一个 hold 想停的位置被吞了。实测 11 份真实 script.json 的 58 个带 hold
       的 beat、61 个 hold：**0 次发生**，所以这条在健康数据上完全不出声。
    2. **hold.at 超出本节点总跨度**：那时「最近的边界」永远是最后一句，等于把留白
       无声地搬到了段尾。上界走 `budget.narration_span_seconds`，也就是
       `script/validate.py` 的 `beat_seconds` **同一份实现**（后者就是它的 Beat 包装），
       所以「把留白放在这段旁白的最后」这种正常创作不触发：实测 61 个真实 hold 一条
       都不报（唯一越过**纯旁白**总长的是 work/saijo E09 的 climax，at=27.0 vs
       26.89 秒，差 0.11 秒，正落在这个宽度里）。

       原来这里是 `offsets[-1] + sum(hold.duration for hold in holds)` —— 一份等价的
       第三实现，注释自己声明「同口径」。那正是 M2 那个「validate 不传 rate」的不一致
       能溜进来的结构原因：两份实现，一份跟着 rate 走、一份不跟，而没有任何东西迫使
       它们一起变。现在只有一份。

       这条不是 validate.py 那道闸的重复：`validate_script()` **只在 script 阶段跑**
       （唯一调用点是 script/single.py），人手改完 `03_script/*.json` 直接
       `--from voice` 时没有任何人再查一遍 at，一个 at=300 照旧一路静默到成片。
    """
    if not sentences:
        return {}
    offsets = sentence_offsets(sentences, rate=rate)
    # 与 script/validate.py 的 beat_seconds 同一份实现：旁白 + 本节点全部留白。
    # 传 `"".join(sentences)` 而不是 offsets[-1]：切句是纯重新分组（拼回去与 strip 过的
    # 原文逐字节相同，见 split_sentences），而 narration_chars 会把全部空白 sub 掉，
    # 所以字数与原 narration 完全一致 —— 但走同一个函数就不会再分叉。
    span = narration_span_seconds("".join(sentences), holds, rate=rate)
    assigned: dict[int, float] = {}
    for hold in holds:
        if warnings is not None and hold.at > span:
            warnings.append(
                f"留白「{hold.quote}」的落点 at={hold.at:.1f} 秒超出本节点跨度 "
                f"{span:.1f} 秒（旁白 + 留白），已贴到最后一句之后"
                "（人工改过 script.json？script 阶段的校验不会再跑一遍）"
            )
        # 平手取靠前的边界：min 遇到相等的 key 保留第一个
        index = min(range(len(offsets)), key=lambda i: abs(offsets[i] - hold.at))
        if warnings is not None and index in assigned:
            warnings.append(
                f"留白「{hold.quote}」（at={hold.at:.1f} 秒）跟前一段留白落到了同一句"
                f"（第 {index + 1} 句）之后，两段静音已合并成 "
                f"{assigned[index] + hold.duration:.1f} 秒"
            )
        assigned[index] = assigned.get(index, 0.0) + hold.duration
    return assigned


def plan_chunks(
    beat: Beat, *, rate: str = DEFAULT_RATE, warnings: list[str] | None = None
) -> list[tuple[str, float]]:
    """把一个 beat 切成 [(要合成的文本, 该 chunk 之后的静音秒数)]。

    `rate` 只影响 hold 落在哪个句边界上（见 sentence_offsets），不影响切句本身。
    `warnings` 原样传给 assign_holds（见那边关于两种坏 hold 的说明）。

    空旁白返回空列表。这条路**只可能**是人工编辑走出来的：LLM 输出侧
    LLMBeat.narration 是 NonBlankStr，空的进不来。调用方（render/tts.py 的
    _plan_pronounceable）负责把它降级成一条 warning。
    """
    sentences = split_sentences(beat.narration)
    if not sentences:
        return []
    hold_after = assign_holds(sentences, beat.audio.holds, rate=rate, warnings=warnings)

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
