"""混音：原声按 timeline 切片拼接后压低，旁白按 offset 延迟叠上去。

只负责拼 ffmpeg 命令行。命令行对不对由测试逐参数断言，ffmpeg 干得对不对是它自己的事。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.atomic import atomic_path
from tenmin.config import DEFAULT_RENDER
from tenmin.intervals import merge_intervals
from tenmin.models import SubtitleCue, Timeline, VoiceTrack
from tenmin.render.ffmpeg import run

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
FULL_VOLUME = 1.0


def duck_gain(duck_db: float) -> float:
    """把 dB 换成线性增益。-12dB ≈ 0.2512。"""
    return 10 ** (duck_db / 20)


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
) -> list[str]:
    """拼出混音用的 ffmpeg 参数列表（不含 ffmpeg 本身）。"""
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法混音")
    if not track.chunks:
        raise ValueError("voice track 里没有任何 chunk，无法混音")
    if len(track.chunks) != len(timeline.narration_offsets):
        raise ValueError(
            f"chunk 数 {len(track.chunks)} 与 narration_offsets 数 "
            f"{len(timeline.narration_offsets)} 不一致，timeline 与 voice 产物不匹配"
        )

    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:a]atrim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"asetpts=PTS-STARTPTS[o{index}]"
        )
    origin_labels = "".join(f"[o{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{origin_labels}concat=n={len(timeline.segments)}:v=0:a=1[orig]")

    expr = duck_volume_expr(timeline.subtitles, duck_gain(duck_db))
    parts.append(f"[orig]volume='{expr}':eval=frame[ducked]")

    chunk_paths: list[str] = []
    for index, (chunk, offset) in enumerate(
        zip(track.chunks, timeline.narration_offsets, strict=True)
    ):
        chunk_paths.append(str(voice_dir / chunk.path))
        parts.append(
            f"[{index + 1}:a]adelay=delays={int(round(offset * 1000))}:all=1[n{index}]"
        )

    if len(track.chunks) == 1:
        voice_label = "[n0]"
    else:
        voice_labels = "".join(f"[n{i}]" for i in range(len(track.chunks)))
        parts.append(f"{voice_labels}amix=inputs={len(track.chunks)}:normalize=0[voice]")
        voice_label = "[voice]"
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
        parts.append(f"anullsrc=r=48000:cl=stereo:d={outro_seconds:.3f}[silence]")
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
    parts.append(f"{final_label}alimiter=limit=1:level=false:latency=true[limited]")
    final_label = "[limited]"

    args = ["-y", "-i", str(video)]
    for path in chunk_paths:
        args.extend(["-i", path])
    args.extend(
        [
            "-filter_complex",
            ";".join(parts),
            "-map",
            final_label,
            "-c:a",
            AUDIO_CODEC,
            "-b:a",
            AUDIO_BITRATE,
            str(out_path),
        ]
    )
    return args


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
    ffmpeg: str = DEFAULT_RENDER.ffmpeg_path,
) -> Path:
    """真跑 ffmpeg 混音，返回产物路径。

    ffmpeg 写的是同目录的 `.part` 文件，跑完才原子改名到 out_path：`-y` 直接写目标
    路径的话，Ctrl-C 或编码中途失败会留下一个 mtime 最新的截断 m4a，而
    pipeline._is_fresh 只比 mtime，下一轮就把它当最新产物跳过、坏音频一路进成片。
    """
    with atomic_path(out_path) as part:
        run(
            build_mix_args(
                video=video,
                timeline=timeline,
                track=track,
                voice_dir=voice_dir,
                out_path=part,
                duck_db=duck_db,
                fade_out_seconds=fade_out_seconds,
                outro_seconds=outro_seconds,
            ),
            ffmpeg=ffmpeg,
        )
    return out_path
