"""时间轴重算。全是纯函数。

对齐规则：clip.start 硬（它过了 v1 validate.py 的 anchor 校正，是有据可查的镜头起点），
clip 时长按 ratio 缩放（clip.end 是 LLM 猜的，只用来算比例）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from tenmin.config import DEFAULT_RENDER, RenderConfig
from tenmin.models import (
    Beat,
    Clip,
    Script,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)
from tenmin.render.chunks import split_sentences
from tenmin.script.budget import narration_chars

# 画面与音频总时长的容忍差。超过就报 warning，不报错。
DRIFT_TOLERANCE = DEFAULT_RENDER.drift_tolerance


def align_to_frame(seconds: float, frame_rate: float | None) -> float:
    """把秒数吸附到最近的帧边界。frame_rate 为 None（帧率未知）时原样返回。

    为什么要对齐：ffmpeg 的 trim 只能按帧切，`trim=start=100.000:end=120.000` 实际
    留下的是 `[ceil(100×fps), ceil(120×fps))` 那些帧，段时长与声明值差最多一帧
    （23.976fps → 41.7ms）。不对齐的话 timeline.json 里记的数字与 ffmpeg 真正用的
    数字不是一回事，而每段的差随机正负，累起来就是「成片时长与 total_seconds 差
    零点几秒」（真实 E02 修前 0.062 秒 = 1.5 帧）。对齐之后段时长恒为整数个帧，
    声明值与实测值就对得上了。

    取**最近**的帧而不是向上/向下取整：偏差上界是半帧而不是一帧，且不会把 clip
    的起点系统性地往后推。

    已知局限（刻意不处理）：这里假设帧落在 `k/fps` 上，而真实源片的视频轨可能有
    非零 start_time（实测 work/saijo 的 mp4 是 0.021333 秒，正好约半帧），那时真实
    帧格在 `start_time + k/fps`。**这不影响每段的帧数**（长度恰好 N/fps 的区间里
    永远正好装 N 个帧格，不管相位），只影响「边界落在两帧中间还是正好压在一帧上」：
    压在帧上时 `%.3f` 的四舍五入（±0.5ms）有可能把首/尾那一帧多算或少算一个，也就
    退回对齐之前的行为。要彻底消掉得把相位也探出来记进产物，收益（最多一帧）不值得
    再加一个字段与一次探测。drift 检查是这件事的兜底。
    """
    if frame_rate is None:
        return seconds
    return math.floor(seconds * frame_rate + 0.5) / frame_rate


def sentence_cues(chunk: VoiceChunk, start: float) -> list[SubtitleCue]:
    """把一个 chunk 的字幕按句切开，按字数比例分配 chunk.duration。

    一个 chunk 常常是好几句话拼起来一次性合成的（省 TTS 调用次数），
    但字幕不能整段话挂几十秒不动——观众读完第一句时，画面上该已经是第二句了。
    没有逐句的真实音频时长，只能按字数比例估算，这是唯一可行的近似。

    这个估算有多准（P2-E B1 实测，不是推理）：拿真 Edge TTS 把 10 集的每一句单独合成
    一遍（359 段），归一化掉逐句合成多出来的静音之后，跟这里的字数比例结果对比 244 个
    句边界 —— 时刻误差 p50 0.365 秒、p90 0.817 秒、max 1.615 秒，>1 秒的 13 个。
    要彻底消掉它得消费 edge-tts 的 SentenceBoundary 元数据（`Communicate.save(音频,
    元数据)` 免费给，实测切句结果与 split_sentences 逐句吻合），但那要改 voice 阶段的
    产物契约（多一个元数据文件、VoiceChunk 加字段、存量 chunk 没有元数据要有退路），
    属于一个独立专项，不在本次范围内。

    **相邻 cue 刻意共享精确边界，不插间隙**（P2-E B3）。ASS 的时间戳只有厘秒精度，
    理论上两条恰好相接的 cue 可能舍入到同一个值、或极短 cue 舍入成零长度。实测 10 集
    已生成的 .ass 共 362 条 Dialogue：`end == start` 0 条、`end < start` 0 条、相邻
    重叠 0 条，298 对相邻 cue 在厘秒级恰好首尾相接（libass 正常处理）。零长度在结构上
    也够不到：cue 时长 ≈ 句字数 × (chunk 时长 / chunk 字数)，而 render/tts.py 的时长
    体检把后者压在 ≈0.11 秒/字以上，所以一个字的句子也有 11 厘秒。
    反过来，插 20–40ms 间隙会让这 298 处**每一处**都多一次字幕闪断，那是真实存在的
    观感损失，换来的是一个测不到的问题。
    """
    sentences = split_sentences(chunk.text)
    if not sentences:
        return [SubtitleCue(start=start, end=start + chunk.duration, text=chunk.text)]
    if len(sentences) == 1:
        return [SubtitleCue(start=start, end=start + chunk.duration, text=sentences[0])]

    weights = [narration_chars(sentence) for sentence in sentences]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1] * len(sentences)
        total_weight = len(sentences)

    cues: list[SubtitleCue] = []
    cursor = start
    for sentence, weight in zip(sentences, weights, strict=True):
        duration = chunk.duration * weight / total_weight
        cues.append(SubtitleCue(start=cursor, end=cursor + duration, text=sentence))
        cursor += duration
    # 累积浮点误差可能让最后一句结束时刻偏离 chunk 边界，钳死到精确值
    cues[-1].end = start + chunk.duration
    return cues


def beat_audio_seconds(chunks: list[VoiceChunk]) -> float:
    """本 beat 的音频总时长。含 hold 静音——静音期间也得有画面。"""
    return sum(chunk.duration + chunk.hold_after for chunk in chunks)


def clip_seconds(clips: Iterable[Clip]) -> float:
    """这些 clip 的总时长。ratio 的分母只能有一处算法。"""
    return sum(clip.duration for clip in clips)


def beat_clip_seconds(beat: Beat) -> float:
    """本 beat **声明**的画面总时长（含那些会被 usable_clips 丢掉的 clip）。"""
    return clip_seconds(beat.clips)


def scale_ratio(audio_seconds: float, clip_seconds: float) -> float:
    if clip_seconds <= 0:
        return 0.0
    return audio_seconds / clip_seconds


def usable_clips(
    beat: Beat, source_duration: float
) -> tuple[list[Clip], list[str]]:
    """挑出「起点本身合法」的 clip，返回 (能用的, warnings)。

    这一遍刻意只查**跟 ratio 无关**的条件，因为 ratio 是拿这些 clip 的时长算出来的：
    一条坏 clip 不该改变它兄弟的缩放倍率。原来是边算边查，于是

    - `start=NaN`（人工改坏 script.json）→ beat_clip_seconds 变成 NaN → ratio 变成
      NaN → **整个 beat 的每一段**都算不出终点，一条不剩；
    - 起点越界的 clip 虽然被丢掉，它的时长仍然算进了 clip_seconds → 幸存的那条被
      按偏小的 ratio 缩放，画面比旁白短一截（实测：真实 E02 的 script 配 900 秒源片，
      画面 143 秒 vs 旁白 214 秒）。

    过滤之后 ratio 只按真的会出画面的 clip 算，上面两条都不成立：坏 clip 只影响自己，
    幸存的 clip 会被拉长到撑满这个 beat 的旁白时长（代价是镜头更慢，但**不失同步**，
    而且丢弃本身已经报了 warning）。
    """
    kept: list[Clip] = []
    warnings: list[str] = []
    for clip in beat.clips:
        # 负数会原样变成 `trim=start=-3.000`（ffmpeg 把它当 0 之前，等于静默改了这一段
        # 的内容），NaN 则**两道**边界比较都为 False，一路漏到 JSON 里变成非法字面量
        # `NaN`。两者都只可能来自人工编辑 script.json —— validate.py 按
        # DialogueTrack.duration 查过 [0, 片长]，但它管不到人手改的那一版。
        if not (math.isfinite(clip.start) and clip.start >= 0):
            warnings.append(
                f"beat {beat.id} 的 clip 起点 {clip.start} 不是 [0, 片长) 里的秒数，"
                "已丢弃该段（人工改过 script.json？）"
            )
            continue
        if clip.start >= source_duration:
            warnings.append(
                f"beat {beat.id} 的 clip 起点 {clip.start:.1f}s 超出源片长 "
                f"{source_duration:.1f}s，已丢弃该段"
            )
            continue
        if not (math.isfinite(clip.duration) and clip.duration > 0):
            warnings.append(
                f"beat {beat.id} 的 clip {clip.start:.1f}s 时长是 {clip.duration}，"
                "已丢弃该段（人工改过 script.json？）"
            )
            continue
        kept.append(clip)
    return kept, warnings


def chunks_by_beat(track: VoiceTrack) -> dict[str, list[VoiceChunk]]:
    grouped: dict[str, list[VoiceChunk]] = {}
    for chunk in track.chunks:
        grouped.setdefault(chunk.beat_id, []).append(chunk)
    return grouped


def build_timeline(
    script: Script,
    track: VoiceTrack,
    source_duration: float,
    *,
    frame_rate: float | None = None,
    cfg: RenderConfig = DEFAULT_RENDER,
) -> tuple[Timeline, list[str]]:
    """按 beat 逐段重算画面时长，产出成片时间轴。

    **一个 beat 只要有 chunk，就必须至少产出一段画面**，否则抛 ValueError。理由：
    音频游标（字幕与旁白落点的唯一驱动）在 clip 检查**之前**就推进了 —— 它必须
    这么推，因为旁白已经合成出来了、占着这段时间。所以一个「有旁白、没画面」的
    beat 会让音频保住时间而画面没有输出，**此后每一段画面都相对旁白整体前移**，
    全片失同步。这不是「尾部少一点画面」，是从那个 beat 起内容全部对错。

    为什么是硬失败而不是本项目惯用的「降级 + warning」：

    - 触发条件本身就是「源片跟剧本对不上」，不是「稿子写得不够好」。实测（真实
      saijo E02 的 script.json + voice.json）把 source_duration 从 1509.97 压到
      900 秒，climax 与 outro 两个 beat 的全部 clip 都落在片长之外，画面比旁白
      少 71.4 秒 —— 真因是「配的 video 不是这一集/被截断了」，唯一有用的动作是
      去修 project.yaml 或换源片，而不是接受一份错位的成片。
    - 降级出来的东西没有价值：填黑场能救回同步，但那个 beat 的旁白就对着黑屏讲，
      而且**同一个原因**还会让相邻 beat 的 clip 被「钳到片尾」，内容照样错。
    - script/validate.py 对**同一个条件**（「某个节点的 clip 全灭」）已经是硬失败
      （见那边 `if not kept` 的长注释）。那道闸按 DialogueTrack.duration（字幕末尾）
      判，这里按**视频**片长判，两者对不上时就漏到这一层 —— 语义上是同一道闸的
      下半段，行为该一致。
    - 代价很小：timeline 是纯计算，贵的 LLM 与 TTS 产物都已经落盘，改完
      project.yaml 跑 `--from timeline` 就继续。

    仍然降级的是**同一个 beat 里丢掉部分 clip**（剩下的还撑得住这段时长）与
    「beat 压根没有 chunk」（音频与画面同时跳过，不会失同步）—— 那两条都不失同步。
    """
    grouped = chunks_by_beat(track)
    segments: list[TimelineSegment] = []
    subtitles: list[SubtitleCue] = []
    offsets: list[float] = []
    warnings: list[str] = []
    audio_cursor = 0.0
    picture_cursor = 0.0

    for beat in script.beats:
        chunks = grouped.get(beat.id, [])
        if not chunks:
            warnings.append(f"beat {beat.id} 没有配音 chunk，已跳过")
            continue

        # 音频游标：字幕与旁白落点都由它驱动
        for chunk in chunks:
            offsets.append(audio_cursor)
            subtitles.extend(sentence_cues(chunk, audio_cursor))
            audio_cursor += chunk.duration + chunk.hold_after

        audio_seconds = beat_audio_seconds(chunks)
        # 先筛掉起点不合法的 clip，再拿**剩下的**算 ratio（理由见 usable_clips）。
        # clip 总时长为 0（clips 为空、或全被筛掉）时 ratio 是 0，于是下面每条 clip 都会
        # 缩放成零长被丢掉，最后由 beat 级的那道闸统一报错 —— 刻意不在这里提前 continue，
        # 免得同一个失同步条件有两条出口、两套消息。
        clips, clip_warnings = usable_clips(beat, source_duration)
        warnings.extend(clip_warnings)
        beat_warnings = list(clip_warnings)
        ratio = scale_ratio(audio_seconds, clip_seconds(clips))
        emitted = 0
        for clip in clips:
            source_start = align_to_frame(clip.start, frame_rate)
            source_end = align_to_frame(source_start + clip.duration * ratio, frame_rate)
            if source_end > source_duration:
                message = (
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 延长到 {source_end:.1f}s "
                    f"超出源片长 {source_duration:.1f}s，已钳到片尾"
                )
                warnings.append(message)
                beat_warnings.append(message)
                # 钳到片尾的这个值刻意**不**再对齐：片尾之后没有下一帧可选，
                # 而往回退一帧会凭空丢掉最后那点画面。
                source_end = source_duration
            if source_end <= source_start:
                message = (
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 缩放后时长为 0，已丢弃该段"
                )
                warnings.append(message)
                beat_warnings.append(message)
                continue
            length = source_end - source_start
            segments.append(
                TimelineSegment(
                    beat_id=beat.id,
                    source_start=source_start,
                    source_end=source_end,
                    timeline_start=picture_cursor,
                    timeline_end=picture_cursor + length,
                )
            )
            picture_cursor += length
            emitted += 1

        if emitted == 0:
            # warnings 在抛异常时是拿不到的（调用方只看得到异常消息），所以把这个
            # beat 自己那几条**拼进消息里** —— 它们才是「为什么一段都没有」的答案。
            detail = "\n".join(f"  - {w}" for w in beat_warnings) or "  - （这个节点没有 clip）"
            raise ValueError(
                f"beat {beat.id} 有 {audio_seconds:.1f}s 旁白却一段画面都没有。"
                f"这会让这个节点之后的画面全部相对旁白前移、整片失同步，"
                f"所以这里直接拦掉。逐条原因：\n{detail}\n"
                f"源片是不是配错集数／被截断了？（project.yaml 的 episodes[].video）"
                f"要么改 03_script/ 里这个节点的 clip 时间戳，再跑 --from timeline。"
            )

    # 这道检查在健康数据上**近乎恒真**（画面长度是从音频长度按 ratio 推出来的，两个
    # 游标按构造必然吻合，实测 10 集残余舍入 ±0.002 秒），它真正的作用是兜住两条降级
    # 路径：clip 被「钳到片尾」而缩短，以及帧对齐带来的半帧级抖动。所以它留着，但
    # 容忍度必须可配（原来写死 0.5）。
    if abs(picture_cursor - audio_cursor) > cfg.drift_tolerance:
        warnings.append(
            f"画面总时长 {picture_cursor:.1f}s 与音频总时长 {audio_cursor:.1f}s "
            f"相差超过 {cfg.drift_tolerance}s，成片尾部会有画面缺失或黑屏，"
            "请检查 timeline.json"
        )

    timeline = Timeline(
        episode=track.episode,
        segments=segments,
        subtitles=subtitles,
        narration_offsets=offsets,
        total_seconds=audio_cursor,
        frame_rate=frame_rate,
    )
    return timeline, warnings
