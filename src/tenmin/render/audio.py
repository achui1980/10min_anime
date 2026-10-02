"""混音：原声按 timeline 切片拼接后压低，旁白按 offset 延迟叠上去。

只负责拼 ffmpeg 命令行。命令行对不对由测试逐参数断言，ffmpeg 干得对不对是它自己的事。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from tenmin.atomic import atomic_path
from tenmin.config import DEFAULT_RENDER, RenderConfig
from tenmin.intervals import merge_intervals
from tenmin.models import HoldWindow, SubtitleCue, Timeline, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter, percent_reporter
from tenmin.render.ffmpeg import FFmpegError, probe_duration, run, run_with_progress
from tenmin.render.timeline import (
    HOLD_EDGE_TOLERANCE,
    HOLD_MIN_COVERAGE,
    hold_coverage,
    source_intervals,
)

AUDIO_CODEC = DEFAULT_RENDER.audio_codec
AUDIO_BITRATE = DEFAULT_RENDER.audio_bitrate
LIMITER_CEILING = DEFAULT_RENDER.limiter_ceiling
FULL_VOLUME = 1.0

# 编码后真峰值体检的容差：超过 loudness_tp + 容差只报 warning，不拒绝发布。
# loudnorm 的 `linear=false`（动态压缩）模式不会把真峰值
# 精确钉在 TP 目标上——它优先保证响度打到 I 目标，真峰值只是个软约束。实测
# akujo E11（真实番剧音轨，AAC 编码）在默认 I=-14/TP=-1.5 下稳定超标约 0.28–0.3dB，
# 且这个超标幅度跟改 loudness_tp（试过 -1.5 到 -3.0）或 loudness_i（试过 -14 到
# -18）基本无关——是这段内容的动态范围特性，不是配置能调走的误差。0.1dB 对真实
# AAC 内容来说太紧，会让这类正常素材被硬拒绝；留够边际但依然能拦住真正离谱的
# 超标（比如 alimiter 配置错到形同虚设的场景）。
_ENCODED_TP_TOLERANCE = 0.5

# 片尾静音的格式。anullsrc 和 concat 前那道 aformat **必须**共用这一组常量：
# 两边各写一份的话，哪天有人只改了一边，concat 就重新退回「靠 ffmpeg 协商」。
#
# 刻意**不**进 RenderConfig（判据见 AGENTS.md：创作旋钮进 config，机械边界留模块级）。
# 它们三个不是「这部番想要什么」，而是「图里两路必须在同一个格式相遇」这条机械要求的
# 一个具体取值 —— 谁改都得两边一起改，而且改错的唯一后果是 concat 退回现场协商，
# 那正是这组常量存在的原因。48k/stereo 是实测所有真实源片的格式（也是它们的上界），
# 44.1k 源会被这条分支重采样到 48k：一个**可查、可测**的下混，比协商出来的那个好。
# 另有一层：ffmpeg 的格式协商结果**跟 filtergraph 的形状有关**（实测：video 侧
# 改成按段输入之后，同一张音频图的 amix 从 48k/stereo 翻成 24k/mono），所以「显式写死」
# 这件事本身就是这里要的东西，把它做成旋钮只会让人以为可以随便调。
SILENCE_SAMPLE_FMT = "fltp"
SILENCE_SAMPLE_RATE = 48000
SILENCE_CHANNEL_LAYOUT = "stereo"


def duck_gain(duck_db: float) -> float:
    """把 dB 换成线性增益。-12dB ≈ 0.2512。"""
    return 10 ** (duck_db / 20)


def mixed_total_seconds(timeline: Timeline, outro_seconds: float) -> float:
    """混音产物有多长。**只是 Timeline.output_seconds 的别名**，保留给现有调用点。

    「多长」的唯一真相住在 models.Timeline.output_seconds（原来这里与 render/video.py
    各算一份同样的加法）：正片是 timeline.total_seconds（build_mix_args 用 apad+atrim
    把它钉死，理由见那边的注释），片尾卡片的静音再接在后面。mix_audio 拿它当进度条的
    总长 —— 进度总长与产物长度必须是同一个数，否则百分比会停在别的地方。
    """
    return timeline.output_seconds(outro_seconds)


def duck_volume_expr(cues: list[SubtitleCue], gain: float) -> str:
    """有旁白的区间压到 gain，其余（留白）回到原声全开。

    没有任何旁白时返回常量 1，让 volume 滤镜变成空操作。

    相接/重叠的 cue 先合并成连续窗，再一个窗一个 between()。原来是一个 cue 一个
    between()，而真实素材里约 87% 的相邻 cue 恰好首尾相接（`前.end == 后.start`
    —— render/timeline.py 的 sentence_cues 按字数比例切句时游标是连续推进的），
    于是几十项里绝大多数只是把同一段连续区间拆成了碎片。真实素材实测
    （work/saijo 10 集）：33→5、29→7、41→6、36→5、42→7、33→6、36→7、34→7、
    38→6、40→8 项，表达式长度从 778~1125 字符降到 154~231。

    合并是**逐点等价**的，不是近似：ffmpeg 的 `between(x,min,max)` 是**闭**区间
    （两端都算命中），所以 `end == 下一条 start` 时两个 between 的并集就是合并后
    那一个；重叠与被包住的情形同理。取 max(end) 而不是最后一条的 end，才不会被
    「短句被长句包住」的 cue 把窗口右界拉回去。用 intervals.merge_intervals 而不是
    自己再写一份区间合并（默认 max_gap=0.0 就是「只合并重叠或首尾相接」）。

    副作用是 volume 的求值成本也降了：eval=frame 意味着这串表达式**每帧**都要算
    一遍，项数少一个数量级等于少一个数量级的 between 调用。
    """
    if not cues:
        return f"{FULL_VOLUME:.4f}"
    windows = "+".join(
        f"between(t,{start:.3f},{end:.3f})"
        for start, end in merge_intervals((cue.start, cue.end) for cue in cues)
    )
    return f"if(gt({windows},0),{gain:.4f},{FULL_VOLUME:.4f})"


# 一份 timeline.json 是人能手改的产物（Timeline 类的 docstring 就写着「改完它跑
# --from audio 就能重出片」），所以 hold_windows 到了混音这一步不能被当成可信输入
# 直接拿去用：混音要施加的单独增益只对 render/timeline.py 的 _assess_holds 已经
# 验过一遍的窗口安全，一份被手改挪动过起止点、或者干脆瞎编出来的窗口可能落在别的
# beat 的画面上、盖住别的旁白、或者根本不对应任何真实的静音间隙。这两个常量与
# hold_coverage/source_intervals 直接复用 render/timeline.py 那一套——「怎么判一个
# 留白窗合法」只能有一份定义，这里要做的是**在混音时**照同一套规则再判一遍，不是
# 发明一套新规则。
_HOLD_BOUNDARY_TOLERANCE = 0.001
_SOURCE_CONTIGUITY_TOLERANCE = 1e-6


def _valid_hold_windows(
    timeline: Timeline, track: VoiceTrack
) -> tuple[list[tuple[int, HoldWindow]], list[str]]:
    """按 segments/narration offsets 的当前实况重新验一遍 timeline.hold_windows。

    返回 `(kept, warnings)`：`kept` 是 `(原始下标, window)` 对的列表，下标是
    window 在 `timeline.hold_windows` 里的位置（不是重新编号的 0..n）—— 后续
    （按每个窗口测出来的响度分配增益）要靠这个下标去查表，编号一变就对不上了。

    这里只负责**判**，不负责修——判不过就跳过单独增益、原样回落到只有 ducking
    的听感，一个字节都不改 `timeline.hold_windows` 本身。
    """
    if len(timeline.narration_offsets) != len(track.chunks):
        return [], ["timeline 与 voice 数量不符，跳过留白窗单独增益"]

    # render/timeline.py 的 _assess_holds 在这一步之前会先判 `ordered_indices ==
    # range(len(chunks))`：narration_offsets 是按 script.beats 遍历顺序追加的，
    # 跟 track.chunks 的原始列表顺序未必一致，beat 交错时两者错位，"正确" 的留白
    # 也可能查到别的 chunk 的音频——那道闸就是防这个。
    #
    # 这里**刻意不**照搬同一道闸：本函数的签名只收 timeline/track（模块 docstring
    # 就是这么设计的——只对着这两份「当前实况」重判，不碰 script），没有
    # script.beats 可用，也不打算为了这一道闸把 script 塞进签名（这次改动的范围
    # 明确排除改 _valid_hold_windows 的公开签名）。退一步讲，没有 script.beats
    # 时，唯一能在这里独立验出的等价条件是「track.chunks 按 beat_id 分组连续且
    # 分组顺序与 narration_offsets 的构造顺序一致」——但分组顺序是否跟
    # script.beats 一致这件事本身就需要 script 才能判，检查不到就是检查不到，
    # 装一个只能防住部分交错情形的「弱闸」只会让人误以为这里已经跟 timeline.py
    # 同口径。真正兜底的是下面 `chunk_starts[i] == window.start` 的**逐毫秒**匹配
    # （容差仅 0.001s）+ 唯一性 + 全部片段连续性/覆盖率/边界检查全部独立成立——
    # 如果 chunk_starts/chunk_ends 因错位而算出一组「garbage」数值，这些独立检查
    # 同时全部凑巧通过的概率可以忽略。更根本的是：beat 交错导致 chunk 列表顺序
    # 与 narration_offsets 不对应，是 build_mix_args 下面 adelay 那段代码本来就有
    # 的同一个假设（同一份 track.chunks/narration_offsets 按位置配对）——不是这次
    # 改动引入的新风险面，属于既有的、跨越整个模块的系统性前提，不该在这一个
    # 函数里单独补一道不完整的闸。
    chunk_starts = list(timeline.narration_offsets)
    chunk_ends = [
        start + chunk.duration
        for start, chunk in zip(chunk_starts, track.chunks, strict=True)
    ]

    kept: list[tuple[int, HoldWindow]] = []
    identities: set[tuple[str, int]] = set()
    warnings: list[str] = []
    order = sorted(
        range(len(timeline.hold_windows)),
        key=lambda i: (timeline.hold_windows[i].start, i),
    )
    for index in order:
        window = timeline.hold_windows[index]
        if _hold_window_is_valid(window, timeline, track, chunk_starts, chunk_ends,
                                  identities, kept):
            kept.append((index, window))
            identities.add((window.beat_id, window.hold_index))
        else:
            warnings.append(
                f"留白窗 {window.beat_id}/{window.hold_index} "
                "与片段/旁白空档不一致，跳过单独增益"
            )
    kept.sort(key=lambda pair: pair[0])
    return kept, warnings


def _hold_window_is_valid(
    window: HoldWindow,
    timeline: Timeline,
    track: VoiceTrack,
    chunk_starts: list[float],
    chunk_ends: list[float],
    identities: set[tuple[str, int]],
    kept: list[tuple[int, HoldWindow]],
) -> bool:
    if not all(
        math.isfinite(v)
        for v in (window.start, window.end, window.source_start, window.source_end)
    ):
        return False
    if (window.beat_id, window.hold_index) in identities:
        return False

    # window.start 必须恰好落在某个「本 beat 的 chunk 结束点」上：这才证明这个
    # 窗口紧跟在一段真实配音之后，是 render/timeline.py 认可的静音间隙的起点，
    # 不是手改挪出来的一个任意时刻。
    boundary_candidates = [
        i for i, chunk in enumerate(track.chunks)
        if chunk.beat_id == window.beat_id
        and abs(chunk_ends[i] - window.start) <= _HOLD_BOUNDARY_TOLERANCE
    ]
    if len(boundary_candidates) != 1:
        return False
    boundary = boundary_candidates[0]

    if (
        boundary + 1 < len(chunk_starts)
        and chunk_starts[boundary + 1] < window.end - _HOLD_BOUNDARY_TOLERANCE
    ):
        return False

    if any(
        window.start < end and start < window.end
        for start, end in zip(chunk_starts, chunk_ends, strict=True)
    ):
        return False

    same_beat_segments = [seg for seg in timeline.segments if seg.beat_id == window.beat_id]
    foreign_segments = [seg for seg in timeline.segments if seg.beat_id != window.beat_id]
    if any(
        window.start < seg.timeline_end and seg.timeline_start < window.end
        for seg in foreign_segments
    ):
        return False

    covering = merge_intervals((seg.timeline_start, seg.timeline_end) for seg in same_beat_segments)
    if not any(lo <= window.start and window.end <= hi for lo, hi in covering):
        return False
    if window.episode != timeline.episode:
        return False

    # 刻意不排序：render/timeline.py 的 _assess_holds 按 segments 的**给定顺序**判连续
    # （画面按时间轴顺序拼接，原片位置本该跟着单调走；倒叙剪辑等非单调重用是真的
    # 不连续）。这里排序会把「先跳回去、再跳过去」这种非单调重用误判成连续，让
    # 混音这一层的再验证比它要复现的那条 continuous 判断（同一函数内的那个 all()
    # 表达式）更松。
    played = source_intervals(same_beat_segments, window.start, window.end)
    if not played:
        return False
    if any(
        abs(played[i + 1][0] - played[i][1]) > _SOURCE_CONTIGUITY_TOLERANCE
        for i in range(len(played) - 1)
    ):
        return False
    if hold_coverage(played, window.source_start, window.source_end) < HOLD_MIN_COVERAGE:
        return False
    if (
        min(lo for lo, _ in played) > window.source_start + HOLD_EDGE_TOLERANCE
        or max(hi for _, hi in played) < window.source_end - HOLD_EDGE_TOLERANCE
    ):
        return False

    if any(window.start < kw.end and kw.start < window.end for _, kw in kept):
        return False

    if not (0 <= window.start < window.end <= timeline.total_seconds):
        return False
    if not (window.source_start < window.source_end):
        return False

    return True


def hold_gain_expr(
    windows: list[tuple[int, HoldWindow]], gains: dict[int, float], fade: float
) -> str:
    """拼一段 `volume=...:eval=frame` 用的增益包络表达式。

    只对 `index in gains` 的窗口生效：窗口内部是 `gains[index]`（dB）换算出的
    线性增益，两侧各花 `fade` 秒线性回落到 1.0（不改变），`fade` 会被夹到不超过
    窗口时长的一半——否则两段渐变会在窗口中点相撞，算出负增益或不连续的跳变。
    窗口之外恒为 1.0。按 `reversed(windows)` 折叠出嵌套的 `if(between(...))`，
    这样列表里靠前的窗口在（理论上不该出现的）重叠场景下优先生效，最终兜底给
    常量 "1"。
    """
    applicable = [(index, window) for index, window in windows if index in gains]
    expr = f"{FULL_VOLUME:g}"
    for index, window in reversed(applicable):
        gain_db = gains[index]
        amp = 10 ** (gain_db / 20)
        fade_eff = min(max(fade, 0.0), (window.end - window.start) / 2)
        if fade_eff > 0:
            fade_in_end = window.start + fade_eff
            fade_out_start = window.end - fade_eff
            envelope = (
                f"if(between(t,{window.start:.3f},{fade_in_end:.3f}),"
                f"{FULL_VOLUME:.6f}+({amp:.6f}-{FULL_VOLUME:.6f})*"
                f"(t-{window.start:.3f})/{fade_eff:.6f},"
                f"if(between(t,{fade_out_start:.3f},{window.end:.3f}),"
                f"{amp:.6f}+({FULL_VOLUME:.6f}-{amp:.6f})*"
                f"(t-{fade_out_start:.3f})/{fade_eff:.6f},"
                f"{amp:.6f}))"
            )
        else:
            envelope = f"{amp:.6f}"
        expr = f"if(between(t,{window.start:.3f},{window.end:.3f}),{envelope},{expr})"
    return expr


# --- 留白窗响度实测：决定「调多少」，不负责决定「哪些窗口可信」 -------------
#
# hold_gain_expr 只管把一份**已经决定好**的 `{index: gain_db}` 拼成 ffmpeg 表达式；
# 「这个窗口的原声该不该调、调几 dB」是这里的事，靠 ebur128 滤镜实测响度决定。
#
# ffmpeg 的 ebur128（`peak=true`）在 stderr 里按 `-nostats` 之外的固有格式打一段
# 汇总报告，形如 `I:         -23.5 LUFS`；`-inf LUFS` 是「这段几乎没有能量」的
# 合法输出（数字静音），不是解析失败，但对我们的用途等价于「测不出」。
_I_LINE = re.compile(r"^\s*I:\s*([-+]?\d+(?:\.\d+)?|-inf)\s*LUFS\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_ebur128_i(stderr: str) -> float | None:
    """从 ebur128 滤镜的 stderr 里取最后一条 Integrated loudness（`I:` 行）。

    取**最后**一条而不是第一条：`peak=true` 时同一次调用的 stderr 里可能夹着别的
    以 `I:` 起头的行（比如 Momentary/Short-term 分段汇总不会用这个前缀，但保守起见
    只信最后出现的这份——那是整段素材测完之后的最终汇总，不是某个中间窗口的快照）。

    一个数字都没解析出来、或者解析出的是 `-inf`（数字静音），一律返回 None 而不是
    `float("-inf")`：调用方（gain_for_window）要用这个值跟 `hold_silence_floor_lufs`
    比大小，`-inf` 参与比较在数学上没问题，但它意味着「这段音频里完全没有可测的能量」，
    跟「量出来是个具体的极低响度」是两件不同的事——前者应该被当成「测不出」，交给
    调用方走「不调、不撒谎」那条分支，而不是被当成一个可信的、极端的响度数字。
    """
    hits = _I_LINE.findall(stderr)
    if not hits:
        return None
    value = float(hits[-1])
    return value if math.isfinite(value) else None


def gain_for_window(
    source_lufs: float | None, voice_lufs: float | None, *, cfg: RenderConfig
) -> tuple[float, str | None]:
    """决定一个留白窗的原声该调几 dB，相对旁白响度的「舒适带」为参照。

    「舒适带」是 `[voice_lufs - hold_relative_lu, voice_lufs + hold_relative_lu]`：
    落在带内不动，落在带外的话只把它拉到**最近的那条边界**（不是拉到 voice_lufs
    本身）——这是刻意留出的余量，原声不必跟旁白一样响，只是不能响得盖过旁白、也
    不能哑得像被切掉了。拉动幅度被 `hold_gain_max_db` 双向夹住，防止一段接近数字
    静音的窗口被硬拉到跟旁白一样响，把底噪/环境声放大成刺耳的一段。

    两处「测不出就不调」是刻意的，且理由不同：
    - `source_lufs` 是 None，或低于 `hold_silence_floor_lufs`（近乎数字静音，
      响度这个量本身在这里已经不可靠——**响度测不出「有没有对白」**，一段被压得
      很低的人声跟真静音在积分响度上可能长得一样）：原声保持原样，不做任何调整，
      理由写进 warning 交给调用方汇总。
    - `voice_lufs` 是 None（旁白参考本身测不出）：没有参照就没有「舒适带」，同样
      不调、同样警告——但消息不同，别把两种不同的「为什么没调」混成一句话。
    """
    if source_lufs is None or source_lufs < cfg.hold_silence_floor_lufs:
        return 0.0, "留白近乎无声或测不出有效响度，保持原声；响度不能证明有对白"
    if voice_lufs is None:
        return 0.0, "旁白参考响度无法测量，跳过留白单窗校准"
    lower = voice_lufs - cfg.hold_relative_lu
    upper = voice_lufs + cfg.hold_relative_lu
    delta = (
        lower - source_lufs
        if source_lufs < lower
        else upper - source_lufs
        if source_lufs > upper
        else 0.0
    )
    return max(-cfg.hold_gain_max_db, min(cfg.hold_gain_max_db, delta)), None


def _origin_parts(timeline: Timeline) -> list[str]:
    """原声那一路的图片段：按 segment 逐段 atrim，再 concat 成一条 `[orig]`。

    这段图与 hold/duck/voice 都无关，是**唯一真相**——build_mix_args 的正片混音
    与 `_meter_args` 的留白窗响度实测都从这同一份代码产生这几行，不能各写一份
    （各写一份的话，量出来的响度可能对应的根本不是真正会被烧进成片的那段声音）。
    """
    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:a]atrim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"asetpts=PTS-STARTPTS[o{index}]"
        )
    origin_labels = "".join(f"[o{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{origin_labels}concat=n={len(timeline.segments)}:v=0:a=1[orig]")
    return parts


def _voice_parts(track: VoiceTrack, offsets: list[float]) -> tuple[list[str], str]:
    """旁白那一路的图片段：每个 chunk 按 offset `adelay`，多于一个再 `amix` 合流。

    跟 `_origin_parts` 一样是**唯一真相**：build_mix_args 的正片混音与
    `_meter_args` 的旁白参考响度实测共用这几行——测的必须是「真正会被叠进成片的
    那份旁白信号」，不能另起一份看起来等价、实则可能漂开的实现。

    只有一个 chunk 时不必 `amix`（`amix=inputs=1` 这种写法在语义上多余，且原来
    `EXPECTED_GRAPH` 就没有它），标签直接是那一路自己的 `[n0]`。
    """
    parts: list[str] = []
    for index, offset in enumerate(offsets):
        parts.append(
            f"[{index + 1}:a]adelay=delays={int(round(offset * 1000))}:all=1[n{index}]"
        )
    if len(track.chunks) == 1:
        return parts, "[n0]"
    voice_labels = "".join(f"[n{i}]" for i in range(len(track.chunks)))
    parts.append(f"{voice_labels}amix=inputs={len(track.chunks)}:normalize=0[voice]")
    return parts, "[voice]"


def _final_mix_parts(
    timeline: Timeline,
    track: VoiceTrack,
    voice_parts: list[str],
    voice_label: str,
    *,
    duck_db: float,
    hold_gains: dict[int, float] | None,
    hold_fade_seconds: float,
    fade_out_seconds: float,
    outro_seconds: float,
    limiter_ceiling: float,
) -> list[str]:
    """剩下那一段：留白窗增益（可选）+ ducking + 与旁白合流 + 定长 + 淡出 +
    片尾静音 + 限幅。

    `voice_parts`/`voice_label` 由调用方传入而不是这里再调 `_voice_parts` —— 旁白
    那一路的图片段必须落在 ducking **之后**、最终 amix **之前**这个具体位置（见
    下面 EXPECTED_GRAPH 锁住的顺序），拆成三个各自独立、互不调用的函数没法表达
    这条「谁嵌在谁中间」的顺序约束，所以由这一层负责把它嵌进正确的位置——
    这样 `_origin_parts`/`_voice_parts` 才能被 `_meter_args` 单独复用而不用
    多算一遍它们并不需要的 ducking/hold 图。
    """
    parts: list[str] = []

    # 留白窗单独增益：只在 hold_gains 非空时插这个节点，且插在 ducking **之前**——
    # ducking 按 subtitle cue 分区（有旁白/没旁白两态），留白窗恰好落在「没旁白」
    # 那一态里，两层滤镜互不冲突，谁先谁后本该没有听感差别；放前面纯粹是为了让
    # `[orig]` 到 `[ducked]` 之间只多一层，不用去改 ducking 自己的输入/输出标签。
    #
    # 没有 hold_gains（默认 None/空 dict）时 duck_input 恒为 "[orig]"，图与
    # 加这层之前逐字节相同——这是硬约束，测试用同一份 EXPECTED_GRAPH 锁住。
    duck_input = "[orig]"
    if hold_gains:
        kept, _ = _valid_hold_windows(timeline, track)
        relevant = [(index, window) for index, window in kept if index in hold_gains]
        if relevant:
            gain_expr = hold_gain_expr(relevant, hold_gains, hold_fade_seconds)
            parts.append(f"[orig]volume='{gain_expr}':eval=frame[holdbalanced]")
            duck_input = "[holdbalanced]"

    expr = duck_volume_expr(timeline.subtitles, duck_gain(duck_db))
    parts.append(f"{duck_input}volume='{expr}':eval=frame[ducked]")

    parts.extend(voice_parts)
    parts.append(f"[ducked]{voice_label}amix=inputs=2:normalize=0[mix]")

    final_label = "[mix]"
    # --- 输出长度：唯一真相是 timeline.total_seconds（+ 片尾卡片） ---
    #
    # 为什么必须显式钉：amix 默认 duration=longest，所以不钉的话输出长度是
    # max(画面音频, 末条配音结束) —— 一个由「哪条输入最长」决定的副产物。render 阶段
    # 用 `-c:a copy -map 1:a` 把这条音轨原样挂进 mp4，而 mp4 的容器时长取两条流的
    # 较大值，于是音频长出一点就静默产出一个更长的 mp4（尾部画面冻结）。
    #
    # 为什么真相是 total_seconds 而不是「画面总长」：total_seconds 就是
    # render/timeline.py 的 audio_cursor，本阶段的 afade 起点（total - fade）和
    # render/video.py 的 fade 起点、进度条总长全部已经以它为准。画面总长
    # （sum(segment 时长)）在 build_timeline 里按构造与它相等，只有 clip 被钳到
    # 片尾/丢弃时才会**变短**，而那条路已经各自报了 warning；让音频跟着一份出过问题
    # 的画面长度走，等于把两个可疑数字绑在一起。
    #
    # 为什么是 apad + atrim 而不是 -t / -shortest：
    # - `-shortest` 钉的是「最短那条输入」，也就是随便某个几秒的旁白 chunk，方向全错。
    # - `-t` 只能截断，补不了「混音比声明时长短」的那一半（画面被钳到片尾时就会短）。
    #   而且片尾静音是在图里 concat 上去的，`-t` 作用在它之后，还得再算一遍总长。
    # apad 负责补齐、atrim 负责截断，两个方向都封死，且落在 afade 之前 —— 淡出的起点
    # 是 timeline 坐标，长度得先对上，淡出才落在该落的地方。
    if timeline.total_seconds > 0:
        parts.append(
            f"{final_label}apad=whole_dur={timeline.total_seconds:.3f},"
            f"atrim=end={timeline.total_seconds:.3f}[mixlen]"
        )
        final_label = "[mixlen]"
    if fade_out_seconds > 0:
        fade_start = max(timeline.total_seconds - fade_out_seconds, 0.0)
        parts.append(
            f"{final_label}afade=t=out:st={fade_start:.3f}:d={fade_out_seconds:.3f}[mixfaded]"
        )
        final_label = "[mixfaded]"
    if outro_seconds > 0:
        # 片尾卡片没有声音，垫一段静音跟视频那边的黑卡对齐。
        #
        # concat 之前先 aformat 把混音这一路显式拍成静音源的格式。图里其余分支都
        # 继承源片格式，只有 anullsrc 是写死的常量，两路在 concat 处相遇时格式由
        # ffmpeg 现场协商 —— 44.1k/mono 的源实测能协商成功（rc=0，不是崩），但
        # 5.1 源片会被**静默下混**成 stereo，没有任何提示。写出来的下混跟协商出来的
        # 下混结果一样，区别是前者可查、可测、可改。
        #
        # 刻意只在这条分支上加：没有片尾静音时图里就没有两路格式相遇的点，那条路
        # 继续继承源片格式（44.1k 源出 44.1k 产物），行为完全不变。
        parts.append(
            f"{final_label}aformat=sample_fmts={SILENCE_SAMPLE_FMT}:"
            f"sample_rates={SILENCE_SAMPLE_RATE}:"
            f"channel_layouts={SILENCE_CHANNEL_LAYOUT}[mixfmt]"
        )
        final_label = "[mixfmt]"
        parts.append(
            f"anullsrc=r={SILENCE_SAMPLE_RATE}:cl={SILENCE_CHANNEL_LAYOUT}:"
            f"d={outro_seconds:.3f}[silence]"
        )
        # concat 之后必须按样本数重建 pts。concat 拼音频时给第二段的偏移是按第一段
        # 「实测到的结束时刻」算的，两段之间会留下一个亚帧级的洞，mp4 muxer 拿这串
        # pts 写 moov 时**随机**少算一截：真实素材实测（work/saijo E02，同一条 argv
        # 连跑 6 遍）容器 duration 在 217.404000 与 214.424229 之间乱跳，偏差正好是
        # 片尾静音那 3 秒。两种结果的 AAC 帧数都是 10192、解码出的样本逐字节相同，
        # 也就是说样本一个没丢，只是 moov 里那个数字写错了 —— 而 render 阶段
        # `-c:a copy` 会把这条音轨连同它的时长声明一起搬进 mp4。
        # asetpts=N/SR/TB 用「已消费样本数 / 采样率」重算每帧的 pts，输出必然连续
        # 单调；实测加上之后 6/6 都是 217.404000，且样本逐字节不变（只动时间戳）。
        parts.append(
            f"{final_label}[silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
        )
        final_label = "[mixfinal]"

    # --- 限幅：把最终信号封在 0 dBFS 以内 ---
    #
    # amix normalize=0 是刻意的（要的就是「原声压低 + 旁白满量程」这个既定听感），
    # 代价是它完全不管相加会不会冲过满刻度。真实素材实测编码前峰值：E01 -4.70、
    # E02 -1.95、E03 -4.26、E06 -0.90、E09 -1.52 dBFS —— 一次都没削波，但最响那集
    # 只剩 0.9dB 余量，换个混响更凶的番、或者把 duck_db 调浅一点就会过线。所以这一层
    # 是**防御性**的，不是在修一个已经发生的 bug。
    #
    # 三个参数缺一不可，任何一个用默认值都会改听感：
    # - limit=1：天花板就是满刻度。峰值没到 1.0 的信号一点增益衰减都不会挨。
    # - level=false：alimiter 的 level 默认 **true**，会把输出自动归一化到 0dB，
    #   等于凭空给整条轨加一次响度变化。
    # - latency=true：alimiter 内部有前瞻缓冲，默认**不**补偿，整条轨会平移一个
    #   attack 窗（默认 5ms）。
    #
    # 为什么必须放在最末尾、在 atrim/afade/concat **之后**：latency=true 补偿延迟的
    # 做法是把输出 pts 往前挪一个前瞻窗，而上面那些滤镜全部按**绝对 timeline 时刻**
    # 工作。实测把它插在 apad/atrim 之前，`atrim=end=214.404` 会少留 239 个样本
    # （≈5ms@48kHz，正好是那个 attack 窗）—— 一个静默的截尾。放在最末尾还顺带符合
    # 「天花板作用在真正出去的那份信号上」这个常规做法。
    #
    # 听感验证（work/saijo E02 + E06，跑完整条 mix graph 出 f32le 原始样本比哈希）：
    # 未削波的素材加上这一层之后样本**逐字节不变**；少 level=false 或少 latency=true
    # 都会变。tests/test_render_audio.py 里有两个 render 标记的用例把这三个参数的
    # 「透明」与「真的限得住」都钉住了。
    parts.append(
        f"{final_label}alimiter=limit={limiter_ceiling:g}:level=false:latency=true[limited]"
    )
    return parts


def _meter_args(
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    *,
    window: HoldWindow | None,
) -> list[str]:
    """拼一段最小的 ffmpeg 参数列表，专门用 `ebur128` 量一段响度，输出到 `-f null -`。

    `window` 给定时量的是**这个留白窗在成片里实际会播放的那段原声**：先走
    `_origin_parts` 把画面按 segment 拼接、落到 timeline 坐标，再从 `[orig]`
    上按 `window.start/end` 截一段——**不能**直接对 `[0:a]` 按 window 的坐标
    trim，那是原片坐标，跟 timeline 坐标只有在没有剪辑重排时才碰巧相等。量的是
    满量程原声（不经过 ducking），因为 ducking 是另一层独立关注点，这里只想知道
    「这段原声本身有多响」。

    `window` 是 None 时量的是**旁白自己的响度参考**：走 `_voice_parts` 把全部
    chunk 按 offset 延迟叠成同一路，再用 `aselect` 只挑出真正落在某个 chunk
    `[offset, offset+duration]` 区间内的样本——排掉 chunk 之间的留白静音间隙，
    否则参考响度会被这些静音稀释，偏离「旁白说话时到底有多响」这个真正要问的问题。

    两条分支的 `-i` 输入顺位都跟 `build_mix_args` 一致（视频永远是 `0`，chunk
    按顺序跟上）——即使旁白分支根本用不到视频输入，也不改这个顺位，理由是
    别让「哪个分支用了哪些输入」变成一件需要对着代码才能确认的事。
    """
    if window is not None:
        parts = [
            *_origin_parts(timeline),
            f"[orig]atrim=start={window.start:.3f}:end={window.end:.3f},"
            f"asetpts=PTS-STARTPTS,ebur128=peak=true[meter]",
        ]
    else:
        voice_parts, voice_label = _voice_parts(track, timeline.narration_offsets)
        speech_windows = "+".join(
            f"between(t,{offset:.3f},{offset + chunk.duration:.3f})"
            for chunk, offset in zip(track.chunks, timeline.narration_offsets, strict=True)
        )
        parts = [
            *voice_parts,
            f"{voice_label}aselect='{speech_windows}',asetpts=N/SR/TB,"
            f"ebur128=peak=true[meter]",
        ]

    args = ["-i", str(video)]
    for chunk in track.chunks:
        args.extend(["-i", str(voice_dir / chunk.path)])
    args.extend(["-filter_complex", ";".join(parts), "-map", "[meter]", "-f", "null", "-"])
    return args


def _measure_hold_gains(
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    windows: list[tuple[int, HoldWindow]],
    *,
    cfg: RenderConfig,
    ffmpeg: str,
) -> tuple[dict[int, float], list[str]]:
    """给每个（已经过 `_valid_hold_windows` 验过的）留白窗量一遍响度，决定增益。

    旁白参考响度**先测一次、只测一次**：没有它就没有 `gain_for_window` 要比的
    「舒适带」，为每个窗口各测一遍原声、再各自套用同一个失败的参照毫无意义——
    既浪费一次 ffmpeg 调用，又会把同一句「测不出参照」的 warning 刷 N 遍。所以
    这里**在进入按窗口的循环之前**就短路：旁白参考测量本身失败（`FFmpegError`）
    或测出来是不可信的值（`_parse_ebur128_i` 返回 None）都直接返回空 dict + 一条
    warning，一个窗口都不碰。

    进入循环之后，**每个窗口的失败只影响它自己**：`run()` 抛 `FFmpegError` 时
    捕获、记一条点名 `beat_id`/`hold_index` 的 warning、`continue` 到下一个窗口——
    不能因为某一个窗口的素材恰好在某个诡异的时间点上让 ffmpeg 不满，就连累其余
    完全独立的窗口也拿不到增益。`gain_for_window` 返回的 warning message（比如
    「近乎无声」）走同样的「跳过这一窗，继续下一个」路径，只是失败原因不同——
    一个是「测量本身失败」，一个是「测量成功但数值不足以支撑调整」，别把两种
    warning 文案混成一句话。
    """
    if not windows:
        return {}, []

    try:
        voice_stderr = run(
            _meter_args(video, timeline, track, voice_dir, window=None), ffmpeg=ffmpeg
        )
    except FFmpegError as error:
        return {}, [f"旁白参考响度测量失败，跳过全部留白窗单独增益：{error}"]
    voice_lufs = _parse_ebur128_i(voice_stderr)
    if voice_lufs is None:
        return {}, ["旁白参考响度无法测量，跳过全部留白窗单独增益"]

    gains: dict[int, float] = {}
    warnings: list[str] = []
    # 按 (window.start, index) 排序：跟 `_valid_hold_windows` 返回 `kept` 时用的
    # 排序键一致，纯粹是为了让警告顺序跟着时间轴走，不影响 `gains` 本身（它是按
    # 原始下标查的 dict，跟遍历顺序无关）。
    for index, window in sorted(windows, key=lambda pair: (pair[1].start, pair[0])):
        try:
            source_stderr = run(
                _meter_args(video, timeline, track, voice_dir, window=window),
                ffmpeg=ffmpeg,
            )
        except FFmpegError as error:
            warnings.append(
                f"留白窗 {window.beat_id}/{window.hold_index} 响度测量失败，"
                f"跳过这一窗的单独增益：{error}"
            )
            continue
        source_lufs = _parse_ebur128_i(source_stderr)
        gain_db, message = gain_for_window(source_lufs, voice_lufs, cfg=cfg)
        if message is not None:
            warnings.append(f"留白窗 {window.beat_id}/{window.hold_index}：{message}")
            continue
        gains[index] = gain_db
    return gains, warnings


def build_mix_args(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    audio_codec: str = AUDIO_CODEC,
    audio_bitrate: str = AUDIO_BITRATE,
    limiter_ceiling: float = LIMITER_CEILING,
    hold_gains: dict[int, float] | None = None,
    hold_fade_seconds: float = 0.1,
) -> list[str]:
    """拼出混音用的 ffmpeg 参数列表（不含 ffmpeg 本身）。

    图本身由三段拼成：`_origin_parts`（原声按 segment 拼接）、`_voice_parts`
    （旁白按 offset 延迟叠加）、`_final_mix_parts`（留白增益 + ducking + 合流 +
    定长 + 淡出 + 片尾静音 + 限幅）。拆成三段是为了让 `_meter_args`（留白窗响度
    实测）能单独复用前两段——测的必须是「真正会被烧进成片的那份原声/旁白」，
    不能另起一份看起来等价、实则可能漂开的图。
    """
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法混音")
    if not track.chunks:
        raise ValueError("voice track 里没有任何 chunk，无法混音")
    if len(track.chunks) != len(timeline.narration_offsets):
        raise ValueError(
            f"chunk 数 {len(track.chunks)} 与 narration_offsets 数 "
            f"{len(timeline.narration_offsets)} 不一致，timeline 与 voice 产物不匹配"
        )

    voice_parts, voice_label = _voice_parts(track, timeline.narration_offsets)
    parts = [
        *_origin_parts(timeline),
        *_final_mix_parts(
            timeline,
            track,
            voice_parts,
            voice_label,
            duck_db=duck_db,
            hold_gains=hold_gains,
            hold_fade_seconds=hold_fade_seconds,
            fade_out_seconds=fade_out_seconds,
            outro_seconds=outro_seconds,
            limiter_ceiling=limiter_ceiling,
        ),
    ]

    args = ["-y", "-i", str(video)]
    for chunk in track.chunks:
        args.extend(["-i", str(voice_dir / chunk.path)])
    args.extend(
        [
            "-filter_complex",
            ";".join(parts),
            "-map",
            "[limited]",
            "-c:a",
            audio_codec,
            "-b:a",
            audio_bitrate,
            str(out_path),
        ]
    )
    return args


# --- 双遍响度归一：把最终成片钉在 cfg.loudness_i/tp/lra 附近 -------------------
#
# alimiter 那一层（build_mix_args 的 `[limited]`）只管「别削波」，从来没管过「整体
# 有多响」——ducking + amix normalize=0 的组合天然产不出稳定的响度（每部番的原声
# 底噪、混响、旁白语速都不一样）。这里在 `[limited]` 之后再挂一层 ffmpeg 官方推荐的
# 两遍 loudnorm 工作流：第一遍只测（`measured=None`，loudnorm 自己分析出
# input_i/tp/lra/thresh + target_offset），第二遍拿这份实测值 + `linear=false`
# 重新过一遍（`linear=false` 才会真的按 true peak 动态压，让 TP 天花板落在配置值
# 附近；`linear=true` 只是线性缩放整条轨，超标的峰值会原样透过去）。
_LOUDNORM_STATS_FIELDS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
_JSON_BLOCK = re.compile(r"\{[^{}]*\}")


def _last_loudnorm_json(stderr: str) -> dict[str, Any] | None:
    """取 stderr 里**最后**一段能解析成 JSON 对象的 `{...}` 块。

    loudnorm 的 `print_format=json` 只在这段结构是扁平的（没有嵌套对象）时才成立
    ——真是这样，所以 `[^{}]*` 这个不含嵌套的字符类就够用，不需要一个真正的括号
    配平解析器。取最后一段是因为同一次调用的 stderr 里可能夹着别的花括号噪音
    （版本横幅、其他滤镜的诊断），而 loudnorm 自己的报告永远是整段跑完之后才打的
    那一份，必然排在最后。
    """
    matches = _JSON_BLOCK.findall(stderr)
    if not matches:
        return None
    try:
        data = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _loudnorm_stats(stderr: str) -> dict[str, float] | None:
    """解析 loudnorm 报告里那五个字段，全部转成 float。

    任何一个字段缺失、不是字符串、转不成 float，或者转出来是非有限值
    （`"-inf"` 是合法的 JSON 取值，但在这个函数眼里跟「读不出」等价——`-inf`
    参与后续「跟配置目标差多少」的减法在数学上没有意义），一律返回 None，逼调用方
    走「拒绝发布」那条分支。`_loudnorm_explicit_silence` 才是「这份 -inf 到底算不算
    一个可以接受的合法状态」的判断，两件事故意分开。
    """
    data = _last_loudnorm_json(stderr)
    if data is None:
        return None
    stats: dict[str, float] = {}
    for field in _LOUDNORM_STATS_FIELDS:
        raw = data.get(field)
        if not isinstance(raw, str):
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        stats[field] = value
    return stats


def _loudnorm_explicit_silence(stderr: str) -> bool:
    """`input_i` 与 `input_tp` **都**字面等于 `"-inf"` 才算「全片确实无声」。

    只有一个是 `-inf` 不算：积分响度测不出但真峰值测得出（或者反过来）意味着
    数据本身不自洽，更可能是别的解析问题，不该被当成「合法的静音」放过去。
    """
    data = _last_loudnorm_json(stderr)
    if data is None:
        return False
    return data.get("input_i") == "-inf" and data.get("input_tp") == "-inf"


def _loudnorm_filter(cfg: RenderConfig, measured: dict[str, float] | None = None) -> str:
    """拼一段 `loudnorm=...` 滤镜串。

    `measured is None` = 第一遍（只测，不整形）；`measured` 给定 = 第二遍，把第一遍
    量出来的五个数喂回去、`linear=false` 让它按 true peak 动态压缩而不是线性缩放
    （线性缩放不会管超标的峰值，动态压缩才会真的把 TP 落到 `cfg.loudness_tp`
    附近）。两遍都带 `print_format=json`：第二遍的报告是 `mix_audio` 编码后体检
    读的那一份，不是白打的。
    """
    base = f"loudnorm=I={cfg.loudness_i:g}:TP={cfg.loudness_tp:g}:LRA={cfg.loudness_lra:g}"
    if measured is None:
        return f"{base}:print_format=json"
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=false:print_format=json"
    )


def _input_identity(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """一份「这些文件现在长什么样」的快照：绝对路径 + 字节数 + mtime（纳秒）。

    两遍响度测量 + 一次真实编码之间隔着好几次 ffmpeg 调用，跨度可能是几分钟。
    如果这期间源视频或某个配音 chunk 被换掉（另一个并发进程在重新合成、用户手动
    换了源片……），第一遍量出来的响度/时长跟真正编码进去的那份素材就不是同一批
    数据——继续发布只会把一份自己都不知道测的是什么的产物送出去。比对内容哈希更
    准，但一部番的源视频动辄几百 MB，逐字节比对的代价跟它要防的事件概率不成比例；
    size + mtime 已经能拦住绝大多数「文件被换过」的场景，且是 O(1) 的 stat 调用。
    """
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def _assert_identity_unchanged(
    sources: list[Path], identity: tuple[tuple[str, int, int], ...], stage: str
) -> None:
    """`mix_audio` 里反复出现的那道闸：这批文件跟开跑前的快照还一样吗？

    四处调用点（测量前、两遍之间、编码期间、成品体检）只有 `stage` 这一句消息
    文案不同，判据与拒绝动作逐字不差——抽出来纯粹是去重，不改变任何一处的
    抛出时机或消息文本的形状（仍然是「混音输入在…发生变化，拒绝发布」，
    `match=r"输入.*变化"` 的既有测试认得出来）。
    """
    if _input_identity(sources) != identity:
        raise FFmpegError(f"混音输入在{stage}发生变化，拒绝发布")


def mix_audio(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    audio_codec: str = AUDIO_CODEC,
    audio_bitrate: str = AUDIO_BITRATE,
    limiter_ceiling: float = LIMITER_CEILING,
    reporter: ProgressReporter | None = None,
    ffmpeg: str = DEFAULT_RENDER.ffmpeg_path,
    cfg: RenderConfig = DEFAULT_RENDER,
    warnings: list[str] | None = None,
    ffprobe: str = DEFAULT_RENDER.ffprobe_path,
) -> Path:
    """真跑 ffmpeg 混音，返回产物路径。

    ffmpeg 写的是同目录的 `.part` 文件，跑完才原子改名到 out_path：`-y` 直接写目标
    路径的话，Ctrl-C 或编码中途失败会留下一个 mtime 最新的截断 m4a，而
    pipeline._is_fresh 只比 mtime，下一轮就把它当最新产物跳过、坏音频一路进成片。

    走 run_with_progress 而不是 run：这是一次几分钟的真实编码（23 段 atrim + 11 路
    adelay + 两级 amix，全程解码整部源片的音轨），原来跑它的时候界面上什么都不动，
    跟卡死没有区别。总长取 timeline.output_seconds —— 跟 build_mix_args 钉住的产物
    长度同源，进度才不会停在别的地方。

    编码前后各挂一道守卫，任何一道触发都直接抛 `FFmpegError`——`atomic_path` 保证
    这不会碰到 `out_path` 一个字节：
    1. 首遍响度测量（`_loudnorm_filter(measured=None)`）读不出五个字段、又不是
       合法的「全静音」，说明这份素材本身或者 ffmpeg 输出格式有问题，不能带着一份
       读不懂的响度报告去编码。
    2. 前后三次 `_input_identity` 快照（测量前、两遍之间、编码后）一旦不一致，
       说明源视频或某个配音 chunk 在这次调用期间被换掉了，两遍测量测的不是同一份
       东西，拒绝发布。
    3. 编码完成后拿**编码产物自己**再measure 一遍：读不出响度报告直接拒绝；真峰值
       超过配置上限（留 `_ENCODED_TP_TOLERANCE` 容差）只报 warning、照常发布（动态
       loudnorm 在大动态范围的真实素材上压不住峰值，硬拒绝会让整集卡在 audio 阶段）；
       实测响度偏离目标超过 1 LUFS 同样只报 warning（alimiter 的天花板有时会压掉
       loudnorm 想要的增益，这是预期的物理限制，不是坏产物）。
    4. 编码产物的实际时长（`probe_duration`）必须跟 timeline 声明的长度对上，否则
       audio 阶段悄悄产出一份被截断/拉长的音轨，而 render 阶段的 `-c:a copy` 会把
       这个错误原样搬进最终 mp4。
    """
    reporter = reporter or NullProgressReporter()
    notices: list[str] = []

    kept, audit_warnings = _valid_hold_windows(timeline, track)
    notices.extend(audit_warnings)

    sources = [video, *(voice_dir / chunk.path for chunk in track.chunks)]
    identity = _input_identity(sources)

    if kept:
        gains, gain_warnings = _measure_hold_gains(
            video, timeline, track, voice_dir, kept, cfg=cfg, ffmpeg=ffmpeg
        )
        notices.extend(gain_warnings)
    else:
        gains = {}

    def full_args(
        dest: str, *, measured: dict[str, float] | None, silent: bool = False
    ) -> list[str]:
        """拼一次完整调用的 argv：`dest` 是真实输出路径，或 `"-"` 表示只测不编码。

        `silent=True`（首遍测量测出「本来就全静音」时的第二遍编码）刻意不追加
        loudnorm——对着全静音信号硬套 loudnorm 只会把底噪或者压缩器的量化噪声放大
        成一段能被听见的东西，不调它才是对的。
        """
        args = build_mix_args(
            video=video,
            timeline=timeline,
            track=track,
            voice_dir=voice_dir,
            out_path=Path(dest),
            duck_db=duck_db,
            fade_out_seconds=fade_out_seconds,
            outro_seconds=outro_seconds,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            limiter_ceiling=limiter_ceiling,
            hold_gains=gains,
            hold_fade_seconds=cfg.hold_fade_seconds,
        )
        if not silent:
            filter_index = args.index("-filter_complex") + 1
            args[filter_index] = (
                f"{args[filter_index]};[limited]{_loudnorm_filter(cfg, measured)}[normalized]"
            )
            map_index = args.index("-map") + 1
            args[map_index] = "[normalized]"
        if dest == "-":
            args = [*args[: args.index("-c:a")], "-f", "null", "-"]
        return args

    _assert_identity_unchanged(sources, identity, "测量期间")

    first_report = run(full_args("-", measured=None), ffmpeg=ffmpeg)
    stats = _loudnorm_stats(first_report)

    _assert_identity_unchanged(sources, identity, "两遍之间")

    if stats is None and not _loudnorm_explicit_silence(first_report):
        raise FFmpegError("首遍响度测量结果缺失或格式错误，拒绝发布")
    if stats is None:
        notices.append("整片无有效响度统计（全静音），按原静音编码")

    with atomic_path(out_path) as part:
        run_with_progress(
            full_args(str(part), measured=stats, silent=stats is None),
            total_seconds=timeline.output_seconds(outro_seconds),
            on_progress=percent_reporter(reporter, "audio"),
            ffmpeg=ffmpeg,
        )

        _assert_identity_unchanged(sources, identity, "编码期间")

        final_report = run(
            ["-hide_banner", "-i", str(part), "-af", _loudnorm_filter(cfg), "-f", "null", "-"],
            ffmpeg=ffmpeg,
        )
        measured_out = _loudnorm_stats(final_report)
        if measured_out is None and not (
            stats is None and _loudnorm_explicit_silence(final_report)
        ):
            raise FFmpegError("编码后响度/真峰值读不出，拒绝发布")

        if measured_out is not None:
            tp_ceiling = cfg.loudness_tp + _ENCODED_TP_TOLERANCE
            if measured_out["input_tp"] > tp_ceiling:
                notices.append(
                    f"成片真峰值实测 {measured_out['input_tp']:.1f} dBTP，"
                    f"超出上限 {tp_ceiling:.1f} dBTP，可能轻微削波；"
                    "可调低 render.loudness_i 后重跑 audio"
                )
            if abs(measured_out["input_i"] - cfg.loudness_i) > 1.0:
                notices.append(
                    f"成片实测 {measured_out['input_i']:.1f} LUFS，"
                    "因峰值限制偏离目标"
                )

        actual_seconds = probe_duration(part, ffprobe=ffprobe)
        expected_seconds = timeline.output_seconds(outro_seconds)
        if abs(actual_seconds - expected_seconds) > 0.1:
            raise FFmpegError(
                f"编码后音轨时长 {actual_seconds:.2f}s 与时间轴不符，拒绝发布"
            )

        _assert_identity_unchanged(sources, identity, "成品体检期间")

    if warnings is not None:
        warnings.extend(notices)
    return out_path
