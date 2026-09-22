"""对白轨从哪来。

三条路，按「无损且便宜」排序：

1. 手传的 SRT —— 人明确指定了，不猜。
2. 视频里的软字幕轨 —— ffmpeg 一条命令抽出来，零成本零误差。**这条分枝的存在是关键**：
   漏掉它会把一个自带字幕轨的片源白白拉去跑几分钟转写，还把质量换低了。
3. 语音转写 —— 只有前两条都不成立时才走，而且要先告诉人一声（它是这三条里唯一一条
   既费时间又有损的）。

三条路都归一成「一个 SRT 文件的路径」，所以下游 build_track 拿到的东西形态完全不变。
顺带的好处：转写结果落成 SRT 就等于缓存，也能被人手动修正。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from tenmin.config import AsrConfig
from tenmin.ingest import asr
from tenmin.render import ffmpeg

# 两份 SRT 的文件名后缀。它们**必须不同**：两条路的复用策略刻意相反（转写那份按 mtime
# 复用、抽出来那份每次重写），共用一个名字既让人分不清手上这份是抽的还是转的，也会让
# 「复用」那半边的判据落到一份不该被信任的文件上。
_ASR_SUFFIX = ".asr.srt"
_EMBEDDED_SUFFIX = ".embedded.srt"


class SubtitleSource(NamedTuple):
    """选中的对白轨来源。

    kind 不是「文件是什么格式」（三条路给出的都是 SRT），而是「这份对白是原生字幕
    还是机器听写的」。下游靠它决定要不要繁转简（日语过 OpenCC 会被改字）、要不要
    跑翻译阶段。
    """

    path: Path
    kind: Literal["srt", "asr"]


def _embedded_dest(cache: Path) -> Path:
    """软字幕轨抽出来的那份 SRT 落在哪。

    跟转写缓存同目录、只换后缀。刻意**不**写成
    `cache.name.replace(".asr.srt", ".embedded.srt")`：那种写法在调用方给的名字不以
    `.asr.srt` 收尾时会**静默什么都不换**，于是两条路撞在同一个文件上 —— 而那正是
    这个函数存在的唯一目的。这里改成「认得出那个后缀就剥掉、认不出就剥掉最后一段
    扩展名」，两种情况都必然换出一个新名字。
    """
    name = cache.name
    stem = name[: -len(_ASR_SUFFIX)] if name.endswith(_ASR_SUFFIX) else Path(name).stem
    return cache.with_name(f"{stem}{_EMBEDDED_SUFFIX}")


def _is_usable_asr_cache(cache: Path, video: Path) -> bool:
    """这份**转写**结果还能用吗。

    0 字节判为不可用：那是上一次被打断留下的残骸，不是产物（跟 pipeline 判断阶段
    新鲜度时的口径一致）。mtime 比源视频旧则判为过期：换了片源（重新压制、换了个
    版本）就得重转。

    只对转写结果成立，**不能**拿去判断抽出来的那份 —— 理由写在
    resolve_subtitle_source 里那段注释。
    """
    if not cache.is_file() or cache.stat().st_size == 0:
        return False
    return cache.stat().st_mtime >= video.stat().st_mtime


def resolve_subtitle_source(
    srt: Path | None,
    video: Path | None,
    *,
    cache: Path,
    asr_config: AsrConfig,
    ffmpeg_path: str = ffmpeg.FFMPEG,
    ffprobe_path: str = ffmpeg.FFPROBE,
) -> SubtitleSource:
    """挑一条路，返回一份可解析的 SRT 及其来源类型。

    cache 是转写结果的落点（调用方给出，通常是 srt/E{NN}.asr.srt）。从软字幕轨抽出来
    的那份走同目录的另一个名字，见 _embedded_dest。

    配置参数叫 `asr_config` 而不是跟转写层一致的 `asr=` —— 本模块顶层 `asr` 这个名字
    已经被导入的模块占了，同名形参会把它在函数体内遮掉。
    """
    if srt is not None:
        if not srt.is_file():
            raise FileNotFoundError(f"找不到字幕文件: {srt}")
        return SubtitleSource(srt, "srt")

    if video is None:
        raise ValueError("既没有字幕文件也没有源视频，无法得到对白轨")
    if not video.is_file():
        raise FileNotFoundError(f"找不到源视频: {video}")

    if ffmpeg.has_subtitle_stream(video, ffprobe=ffprobe_path):
        # 这一路**每次都重抽**，绝不拿「文件在 + mtime 够新」当复用判据 —— 跟下面转写
        # 那一路的处置刻意相反。差别的来源是「被打断时留下什么」：
        #
        # - extract_subtitle_track 是 ffmpeg 直接流式写 dest（它的 docstring 明确把
        #   「不要拿 dest 存在当复用判据」这件事转交给调用方）。中途 Ctrl-C 留下的半份
        #   文件**语法合法** —— SRT 没有文件尾结构，一串顺序 cue 块的前缀本身就是一份
        #   能解析的 SRT，所以 srt_parser 会零警告地把它解析成半份对白轨。
        # - transcribe 走 tenmin.atomic 落盘，失败路径上一个文件都不留，所以那边
        #   「文件在就复用」是安全的。
        #
        # 另一条可选做法是在这里包 atomic_path + 完整性体检，然后照样按 mtime 复用。
        # 没这么做的理由是「体检」这一半找不到实现：截断的 SRT 跟完整的 SRT 在结构上
        # 无从区分（前缀即合法），体检只能退化成「非空」，而那恰好是挡不住这个问题的
        # 判据。剩下 atomic_path 那一半能换到的收益只是省掉一次 demux —— 而转写省掉的
        # 是数分钟（倍率见 ingest/asr.py 里记的那次 spike），两者不在同一个量级。所以
        # 这条路选最笨也最难错的做法。
        embedded = _embedded_dest(cache)
        ffmpeg.extract_subtitle_track(video, embedded, ffmpeg=ffmpeg_path)
        return SubtitleSource(embedded, "srt")

    if _is_usable_asr_cache(cache, video):
        return SubtitleSource(cache, "asr")

    # 这是三条路里唯一一条既费时间又有损的，所以让它被看见。刻意不做成一个要用户每次
    # 记得传的 flag：软字幕那条是无损的，不值得为它多打字；值得被看见的只有这一条。
    print(f"  {video.name} 没有字幕轨，将对音轨做语音转写（约需数分钟）")
    asr.transcribe(video, cache, asr=asr_config, ffmpeg_path=ffmpeg_path)
    return SubtitleSource(cache, "asr")
