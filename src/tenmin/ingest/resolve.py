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

from tenmin.config import DEFAULT_RENDER, AsrConfig
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
    """软字幕轨抽出来的那份 SRT 落在哪。跟转写缓存同目录、只换后缀。

    **防撞车靠的是 `Path(name).stem`**：它剥掉最后一段扩展名，所以拼回
    `.embedded.srt` 之后对**任何**输入都必然得到一个跟 cache 不同的名字（`E11`
    → `E11.embedded.srt`，连 `E11.embedded.srt` → `E11.embedded.embedded.srt`
    这种自指输入也换得出来）。这一条是有意义的不变量，别把它换成
    `cache.name.replace(_ASR_SUFFIX, _EMBEDDED_SUFFIX)`：那种写法只在名字恰好以
    `.asr.srt` 收尾时有效，其余情况**静默原样返回** —— 实测 6 个形态（`E11.asr.srt` /
    `E11.srt` / `E11.asr.SRT` / `dialogue.srt` / `E11` / `E11.embedded.srt`）里有 5 个
    会直接撞回 cache 自己，只有规范的那个 `E11.asr.srt` 例外，而两条路共用一个文件正是
    这个函数存在的唯一目的所要避免的事。

    `endswith(_ASR_SUFFIX)` 那个分枝**只为观感，不参与防撞车**：没有它
    `E11.asr.srt` 会变成 `E11.asr.embedded.srt`（不撞车，只是名字里留着一个已经
    不成立的 `.asr`）。真想删就连
    test_the_canonical_cache_name_yields_a_clean_embedded_name 一起删 —— 那条测试
    专门钉这个观感结果，参数化那条不变量测试对它无感。
    """
    name = cache.name
    stem = name[: -len(_ASR_SUFFIX)] if name.endswith(_ASR_SUFFIX) else Path(name).stem
    return cache.with_name(f"{stem}{_EMBEDDED_SUFFIX}")


def _is_usable_asr_cache(cache: Path, video: Path) -> bool:
    """这份**转写**结果还能用吗。

    0 字节判为不可用：那是上一次被打断留下的残骸，不是产物（跟 pipeline 判断阶段
    新鲜度时的口径一致）。mtime 比源视频旧则判为过期：换了片源（重新压制、换了个
    版本）就得重转。

    **刻意不把 AsrConfig 掺进判据**（不按模型/语言的指纹失效）。理由是这份 `.asr.srt`
    是一份**人能手改**的产物 —— 转差了直接改文件、下次本函数就按 mtime 认它（手改把
    mtime 推过了源视频），照样按 `kind="asr"` 复用，这是转写层落盘成 SRT 而不是直接
    返回 cue 的主要好处之一（见 ingest/asr.py 的模块 docstring）。注意手改**不会**让它
    变成「手传 SRT」那条分枝：那条只看调用方传进来的 `srt`，而改一份缓存文件不会往
    project.yaml 的 `episodes[].srt` 里放任何东西。这正是想要的 —— 手工修过的日语听写
    仍然是听写，`kind` 决定的繁转简/翻译处置不该因为有人动过文件就变。
    按指纹失效就意味着用户改完 `asr.model` 之后那些手改会被**静默冲掉**，
    而那比反过来那个毛病坏得多：「换了模型却没重转」打开文件就看得出来（也能靠删文件
    解决），静默覆盖是无声的。代价说清楚：换模型想重转必须自己删掉那份 `.asr.srt`。

    只对转写结果成立，**不能**拿去判断抽出来的那份 —— 理由写在
    resolve_subtitle_source 里那段注释。
    """
    if not cache.is_file():
        return False
    # 一次 stat 取两个字段。分两次调用不只是多一次 syscall，还会让两个判据看到**两个
    # 不同时刻**的文件状态。
    info = cache.stat()
    if info.st_size == 0:
        return False
    return info.st_mtime >= video.stat().st_mtime


def resolve_subtitle_source(
    srt: Path | None,
    video: Path | None,
    *,
    cache: Path,
    asr_config: AsrConfig,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
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
        #   文件**解析器不会报错**，只是对白少了一半 —— SRT 没有文件尾结构，一串顺序
        #   cue 块的前缀本身就是一份能解析的 SRT。
        # - transcribe 走 tenmin.atomic 落盘，失败路径上一个文件都不留，所以那边
        #   「文件在就复用」是安全的。
        #
        # 关键**不是**「一条警告都没有」。实测拿一份 5 条 cue 的 SRT 穷举全部 208 个
        # 截断点，其中 140 个（67%）会多出至少一条 skipped_blocks —— 截断落在序号行或
        # 时间戳行中间时，那半个块找不到 `-->` 就被整块跳过；剩下 68 个（33%，截断落在
        # 正文里或正好在块边界）连这条警告都没有。而「零警告」**不等于「只丢一点」**：
        # 那 68 个里有 14 个只解析出 1 条 cue（5 条丢了 4 条），零警告截断点的 cue 数是
        # 1..5 全谱。也就是说损失最惨的那一档恰好落在一声不响的那一边。真正的问题是
        # **那条警告指不出真因**：skipped_blocks 跟「片源字幕格式略歪」完全同形，而它
        # 恰好是报给用户看片源质量的那个数字，多出来的一条会被当成片源的毛病。
        #
        # 上面三个数里只有 140 有内容，别把 67%/33% 当成 SRT 截断的普适比例：
        # TIMESTAMP_PATTERN 末尾的 `(?!\d)` 让毫秒组 1–3 位都可匹配，所以 29 字时间戳行
        # 的最小可匹配前缀是 27 字，每块开头那 len(序号)+27 个截断点必然都找不到 `-->`
        # → 5 块 × 28 = 140，与正文长度**无关**（实测正文 3/9/20 字时截断点总数分别
        # 184/214/269，有警告的恒是 140；把序号换成两位、三位则每块变 29、30）。截断点
        # 总数与零警告个数反过来只是素材长度的算术函数，换一份素材那两个百分比就变。
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

    # 这是三条路里唯一一条既费时间又有损的，所以让它被看见，而且必须打在调用**之前**
    # （几分钟的静默会让人以为卡死了）。刻意不做成一个要用户每次记得传的 flag：软字幕
    # 那条是无损的，不值得为它多打字；值得被看见的只有这一条。
    #
    # 这一行的增量信息只有「为什么走到了这一步」。耗时提示留给 transcribe 自己那句
    # 「正在转写音轨（约需数分钟）」—— 两行都带文件名又都带「约需数分钟」，第二遍就
    # 只是噪音。
    print(f"  {video.name} 没有字幕轨，只能走语音转写")
    asr.transcribe(video, cache, asr=asr_config, ffmpeg_path=ffmpeg_path)
    return SubtitleSource(cache, "asr")
