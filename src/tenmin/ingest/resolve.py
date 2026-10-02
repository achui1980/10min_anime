"""对白轨从哪来。

四条路，按「无损且便宜」排序：

1. 手传的 SRT —— 人明确指定了，不猜。
2. 视频里的软字幕轨 —— ffmpeg 一条命令抽出来，零成本零误差。**这条分枝的存在是关键**：
   漏掉它会把一个自带字幕轨的片源白白拉去跑几分钟转写，还把质量换低了。声明了硬字幕的
   片源也照样先走这条：文本字幕轨是最准的素材。
3. 画面 OCR —— 只在 project.yaml 声明了「这部番带硬字幕」时才走（不做自动探测）。画面上
   那份是人工翻译好的中文字幕，比听写好得多。OCR 跑不了（没装 extra、不是 macOS）时
   **直接报错，不回落到语音转写**：用户明确说了要用画面上那份更好的素材。
4. 语音转写 —— 只有前三条都不成立时才走，而且要先告诉人一声。

四条路都归一成「一个 SRT 文件的路径」，所以下游 build_track 拿到的东西形态完全不变。
顺带的好处：OCR 与转写的结果落成 SRT 就等于缓存，也能被人手动修正。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from tenmin.config import DEFAULT_OCR, DEFAULT_RENDER, AsrConfig, OcrConfig
from tenmin.ingest import asr, ocr
from tenmin.render import ffmpeg

# 三份 SRT 的文件名后缀。它们**必须两两不同**：抽出来那份每次重写，OCR 与转写那两份按
# mtime 复用，共用一个名字既让人分不清手上这份是怎么来的，也会让「复用」那半边的判据落到
# 一份不该被信任的文件上。OCR 那份的路径由调用方给（pipeline.Paths.ocr_cache 是那个名字的
# 唯一权威），这里的后缀只是同一个约定的另一份记录，由
# tests/test_pipeline.py 的 test_the_ocr_cache_name_ends_with_ocr_srt 核对两边一致。
_ASR_SUFFIX = ".asr.srt"
_EMBEDDED_SUFFIX = ".embedded.srt"
_OCR_SUFFIX = ".ocr.srt"

# ffmpeg 拒绝「位图字幕 → 文本字幕」时 stderr 里的原话（实测 ffmpeg 9.0.1 的二进制里
# 就是这一句）。认它是为了把一整屏 ffmpeg 报错换成一句能照着做的中文。
#
# 只挑句子中间那一段来匹配，不含首尾：前缀 `Subtitle encoding currently only possible`
# 里带 "currently"，那是 ffmpeg 留给自己改口的措辞；而整句连标点一起钉住的话，任何一次
# 上游润色都会让这个分枝静默失效（失效的表现是退回旧行为 —— 一屏英文报错，不是变红，
# 所以测试抓不到）。中间这段描述的是 ffmpeg 的能力边界本身，最不容易被改。
_BITMAP_SUBTITLE_MARKER = "text to text or bitmap to bitmap"


class SubtitleSource(NamedTuple):
    """选中的对白轨来源。

    kind 不是「文件是什么格式」（四条路给出的都是 SRT），而是「这份对白是怎么来的」：
    原生字幕（srt）、画面 OCR（ocr）、机器听写（asr）。下游靠它决定要不要繁转简（srt 与
    ocr 转，日语听写过 OpenCC 会被改字所以不转）、translate 阶段怎么处置（srt 跳过、
    ocr 不调 LLM 直接交付简体字幕、asr 翻译）。
    """

    path: Path
    kind: Literal["srt", "asr", "ocr"]


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


def _is_usable_cache(cache: Path, video: Path) -> bool:
    """这份**转写或 OCR** 结果还能用吗。两者共用同一个判据，下面以转写为例说明。

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
    OCR 那份 `.ocr.srt` 同理：**刻意不看 OcrConfig**，改了 crop_top / similarity 之类想
    重认，得自己删掉它。

    只对转写与 OCR 的结果成立，**不能**拿去判断抽出来的那份 —— 理由写在
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
    hardsub: bool = False,
    ocr_cache: Path | None = None,
    ocr_config: OcrConfig = DEFAULT_OCR,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> SubtitleSource:
    """挑一条路，返回一份可解析的 SRT 及其来源类型。

    cache 是转写结果的落点（调用方给出，通常是 srt/E{NN}.asr.srt）。从软字幕轨抽出来
    的那份走同目录的另一个名字，见 _embedded_dest。ocr_cache 是 OCR 结果的落点（通常是
    srt/E{NN}.ocr.srt），只在 hardsub 为真时才用得上、也才必须给。

    hardsub 是「这一集声明了硬字幕」（调用方用 ProjectConfig.hardsub_enabled 算好再传），
    这一层不读 project.yaml。

    配置参数叫 `asr_config` / `ocr_config` 而不是跟下层一致的 `asr=` / `ocr=` —— 本模块
    顶层 `asr`、`ocr` 这两个名字已经被导入的模块占了，同名形参会把它们在函数体内遮掉。
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
        try:
            ffmpeg.extract_subtitle_track(video, embedded, ffmpeg=ffmpeg_path)
        except ffmpeg.FFmpegError as exc:
            # 位图字幕轨（Blu-ray PGS / DVD VobSub）会走到这里：`has_subtitle_stream`
            # 只回答「有没有字幕轨」，答不了「是不是文本」，于是抽取那一步撞上 ffmpeg 的
            # 跨族转码守卫。**刻意不在这里自动回落到语音转写**：那等于把一次几分钟的
            # 有损操作藏在一个看起来只是「抽个字幕」的分枝里，而且位图轨是能 OCR 的
            # （信息还在），悄悄换成听写是把用户手上更好的那份素材丢了。
            #
            # 也刻意不把这段判断挪进 ffmpeg.py：那一层的 docstring 明写「不含业务判断」，
            # 而「碰上位图轨该怎么办」（手传 SRT？换片源走听写？）是本模块的四岔职责。
            if _BITMAP_SUBTITLE_MARKER not in str(exc):
                raise
            # 声明了硬字幕时那句「换没有字幕轨的片源走语音转写」是错的指路：没有字幕轨的
            # 片源在这种配置下走的是画面 OCR，不是听写。行为不变（照样不自动换路），只换
            # 消息，把真实的两条出路说清楚。
            if hardsub:
                raise ValueError(
                    f"{video.name} 带着一条位图字幕轨（PGS / VobSub），软字幕轨那一岔优先，"
                    "所以声明了硬字幕也没有走到画面 OCR。手传一份 --srt，"
                    "或者把片源里的字幕轨去掉，让它走画面 OCR。"
                ) from exc
            raise ValueError(
                f"{video.name} 的字幕轨是位图格式（PGS / VobSub），抽不成 SRT。"
                "手传一份 --srt，或者用一个没有字幕轨的片源让它走语音转写。"
            ) from exc
        return SubtitleSource(embedded, "srt")

    if hardsub:
        if ocr_cache is None:
            raise ValueError(f"{video.name} 声明了硬字幕，但调用方没给 OCR 结果的落点（ocr_cache）")
        if _is_usable_cache(ocr_cache, video):
            return SubtitleSource(ocr_cache, "ocr")
        # 耗时告知由 recognize 自己打（那一行已经带着「声明了硬字幕」这个理由和预估分钟数），
        # 这里不再重复一遍。OCR 失败（含没装 extra）原样往外抛，不往下落到语音转写。
        ocr.recognize(
            video,
            ocr_cache,
            ocr=ocr_config,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )
        return SubtitleSource(ocr_cache, "ocr")

    if _is_usable_cache(cache, video):
        return SubtitleSource(cache, "asr")

    # 这是剩下几条路里唯一一条既费时间又有损的，所以让它被看见，而且必须打在调用**之前**
    # （几分钟的静默会让人以为卡死了）。刻意不做成一个要用户每次记得传的 flag：软字幕
    # 那条是无损的，不值得为它多打字；值得被看见的只有这一条。
    #
    # 这一行的增量信息只有「为什么走到了这一步」。耗时提示留给 transcribe 自己那句
    # 「正在转写音轨（约需数分钟）」—— 两行都带文件名又都带「约需数分钟」，第二遍就
    # 只是噪音。
    print(f"  {video.name} 没有字幕轨，只能走语音转写")
    asr.transcribe(video, cache, asr=asr_config, ffmpeg_path=ffmpeg_path)
    return SubtitleSource(cache, "asr")
