"""TTS。跟 v1 的 LLMProvider 完全同构：真实引擎藏在 Protocol 后面，测试用假引擎。"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path
from typing import Protocol, runtime_checkable

from tenmin.config import DEFAULT_RENDER, RenderConfig
from tenmin.models import Beat, Script, VoiceChunk, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.chunks import is_pronounceable, plan_chunks
from tenmin.render.ffmpeg import FFPROBE, probe_duration
from tenmin.script.budget import DEFAULT_RATE, narration_chars, narration_seconds

# speed_factor 历史上住在本模块（叫 _speed_factor），现在唯一实现在 script/budget.py
# ——预算估算与本模块的时长体检必须共用同一份 rate 解析。这里原样 re-export，是为了让
# 「从 render.tts 拿 speed_factor」这个既有调用面继续成立。
from tenmin.script.budget import speed_factor as speed_factor  # noqa: PLC0414

TTS_MAX_ATTEMPTS = 3

# chunk 文件名里那段哈希的长度。8 个 hex = 32 bit：单集一百多个 chunk 的碰撞概率
# ~1e-5 量级，而更长只会让文件名更难人眼扫。
CHUNK_HASH_LENGTH = 8

# --- 合成结果的时长体检。依据全写在 _duration_bounds 的 docstring 里 ---
# 刻意不做成 RenderConfig 旋钮：它不是「这部番想要多长的解说」那类创作参数，而是
# 「Edge TTS 物理上不可能吐出这种音频」的合法性边界，标定自 115 个真实 chunk。
DURATION_TOLERANCE = 0.5
DURATION_ABSOLUTE_SLACK = 2.0

# --- 重试退避。参数与命名一律沿用 script/llm.py 的既有模式 ---
# 基数 1 秒：Edge TTS 的瞬时 403 / WebSocket 握手失败的恢复窗是秒级。
# 上限 30 秒：单集有上百个 chunk，单次等待再往上翻只会把一次注定失败的运行拖成十几分钟。
# 抖动是**乘性**的 [1, 1.25)：只会加不会减，同时打散批量模式下多集连着撞同一个限流窗。
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
BACKOFF_JITTER_RATIO = 0.25

# 「重试一定没用」的异常。edge_tts.Communicate.__init__ 与 data_classes.TTSConfig
# .__post_init__ 对 voice / rate / volume / pitch / proxy / 超时参数做的全是纯参数校验
# （communicate.py:346/355/360-363 抛 TypeError，data_classes.py:38 的
# validate_string_param 抛 ValueError，例如 rate 不匹配 ^[+-]\d+%$ 就是
# "Invalid rate '+abc%'."）。重试层每次都传同一组参数进去，所以它们必然以同样的方式再炸
# 一次 —— 背靠背发三遍只是把用户看到报错的时间推后几秒。
_PERMANENT_ERRORS = (TypeError, ValueError)


async def _sleep(seconds: float) -> None:
    """退避用的 sleep。**刻意做成模块级函数**，测试 monkeypatch 掉它就既不真睡、
    又能把整条退避序列的时长逐个断言出来。"""
    await asyncio.sleep(seconds)


def _rand() -> float:
    """[0, 1) 的抖动源。同样是模块级函数，为的是让退避序列在测试里可确定。"""
    return random.random()


def _backoff_delay(attempt: int) -> float:
    """第 attempt 次尝试失败后要等多久（attempt 从 1 开始）。"""
    delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
    return delay * (1.0 + BACKOFF_JITTER_RATIO * _rand())


class TTSError(RuntimeError):
    """TTS 层已经自带一句人话的异常。进了 cli.PIPELINE_ERRORS，所以只印消息不印 traceback。"""


@runtime_checkable
class TTSEngine(Protocol):
    @property
    def fingerprint(self) -> str:
        """「这台引擎会出什么音色」的身份串。进 chunk 文件名的哈希，所以 voice / rate
        一改，磁盘上的旧 chunk 就自动失效，不会拿旧音色的音频冒充新配置。"""
        ...

    async def synthesize(self, text: str, out_path: Path) -> float:
        """合成一段语音，返回真实时长（秒）。"""
        ...


def _expected_seconds(text: str, rate: str) -> float:
    """这段文本「应该」有多长。**跟时长预算共用同一个估算器**。

    原来这里有一份自己的 rate 解析（`_speed_factor`）与自己的 chars/CPS/factor 算式，
    而 script/budget.py 那边压根不看 rate —— 两份实现随时会分叉。现在两边都走
    budget.narration_seconds，`speed_factor` 也只有那一份（本模块 re-export 它，
    老调用点与测试照旧能从 render.tts 拿到）。
    """
    return narration_seconds(text, rate=rate)


def _duration_bounds(text: str, rate: str) -> tuple[float, float]:
    """合成结果的合理时长区间。落在区间外一律当「这个文件不能用」。

    依据（实测，不是拍脑袋）：work/ 下 10 集共 115 个真实 chunk 的
    `实测时长 / (字数 / 4.5)` 落在 **0.809 – 1.179**（中位数 0.926）。所以
    ±50% 的相对带宽相当于给实测下界留了 1.6 倍、上界留了 1.27 倍的余量。

    再叠一个 2 秒的**绝对**余量，因为纯比例带宽会误杀短句：Edge TTS 每段音频首尾都带
    固定的静音开销，一个 2 字的 chunk 估算只有 0.44 秒、实测能到 1.2 秒（比值 2.7），
    纯 ±50% 会把它判死。实测最短的 chunk 是 9 字 / 2.16 秒，加了绝对余量后它距上界还
    有 2.8 秒；把这个 band 套回全部 115 个真实 chunk，**零误杀**，最紧的一侧余量
    2.16 秒。

    换句话说：这条体检只拦「零时长」「只出了几分之一就断流」「挂了几分钟吐出一堆静音」
    这类物理上不可能是正常语音的结果，正常创作与正常网络抖动一律放过。

    已知边界：4.5 字/秒是**中文**的语速，一个 60% 以上是拉丁字母的 chunk（英文读得比
    中文快得多）会撞下界而被判死。旁白是中文，实测 115 个 chunk 里没有一个接近这种形态，
    所以接受这个风险 —— 而且它的失败是响亮的（报错直接印出字数、rate 与区间），不是静默
    地用一个坏文件出片。
    """
    expected = _expected_seconds(text, rate)
    lower = max(0.0, expected * (1.0 - DURATION_TOLERANCE) - DURATION_ABSOLUTE_SLACK)
    upper = expected * (1.0 + DURATION_TOLERANCE) + DURATION_ABSOLUTE_SLACK
    return lower, upper


class EdgeTTSEngine:
    """Edge TTS。合成先落 `.part` 再原子改名，所以正式文件永远是体检过的。"""

    def __init__(
        self,
        voice: str = "zh-CN-YunxiNeural",
        rate: str = "+0%",
        *,
        proxy: str | None = None,
        connect_timeout: int = 10,
        receive_timeout: int = 60,
        chunk_timeout_seconds: float = 300.0,
        ffprobe: str = DEFAULT_RENDER.ffprobe_path,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.proxy = proxy
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.chunk_timeout_seconds = chunk_timeout_seconds
        self.ffprobe = ffprobe

    @property
    def fingerprint(self) -> str:
        """只含真正改变输出音频的参数。proxy / 超时改了不影响音色，不该让缓存失效。"""
        return f"edge|{self.voice}|{self.rate}"

    async def synthesize(self, text: str, out_path: Path) -> float:
        import edge_tts

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # edge_tts.Communicate.save() 内部是 `open(audio_fname, "wb")` 的**流式**写
        # （communicate.py:616）。Ctrl-C 或断网留下的是一个「非空但截断」的 mp3，
        # 而复用判据只看 st_size > 0 —— 它偏短的时长会静默扭曲此后全部时间轴偏移。
        # 所以合成一律落临时文件，体检通过才 os.replace 到正式名字（同目录内的
        # rename 在 POSIX 上是原子的）。
        part_path = out_path.with_name(out_path.name + ".part")
        try:
            try:
                async with asyncio.timeout(self.chunk_timeout_seconds):
                    communicate = edge_tts.Communicate(
                        text,
                        self.voice,
                        rate=self.rate,
                        proxy=self.proxy,
                        connect_timeout=self.connect_timeout,
                        receive_timeout=self.receive_timeout,
                    )
                    await communicate.save(str(part_path))
            except TimeoutError as error:
                # asyncio.timeout 到点抛的是内置 TimeoutError。edge-tts 内部的
                # sock_read 超时是 aiohttp 的异常，两者不会互相误吞。
                raise TTSError(
                    f"单个 chunk 合成超时（超过总时长上限 {self.chunk_timeout_seconds} 秒）："
                    f"{text!r}。确实需要更久的话调高 render.tts_chunk_timeout_seconds。"
                ) from error
            # 阻塞的 subprocess，必须扔到线程里：直接 await 不了，直接调会把事件循环
            # 整个卡住（synthesize_track 的并发就完全白做）。
            duration = await asyncio.to_thread(probe_duration, part_path, ffprobe=self.ffprobe)
            lower, upper = _duration_bounds(text, self.rate)
            if duration <= 0 or not (lower <= duration <= upper):
                raise TTSError(
                    f"合成结果时长 {duration:.3f} 秒不在合理区间 "
                    f"[{lower:.3f}, {upper:.3f}] 内（{narration_chars(text)} 字，"
                    f"rate={self.rate}），大概率是流被截断或只出了静音：{text!r}"
                )
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        part_path.replace(out_path)
        return duration


def build_tts_engine(cfg: RenderConfig) -> TTSEngine:
    return EdgeTTSEngine(
        voice=cfg.voice,
        rate=cfg.rate,
        proxy=cfg.tts_proxy,
        connect_timeout=cfg.tts_connect_timeout,
        receive_timeout=cfg.tts_receive_timeout,
        chunk_timeout_seconds=cfg.tts_chunk_timeout_seconds,
        ffprobe=cfg.ffprobe_path,
    )


async def synthesize_with_retry(
    engine: TTSEngine,
    text: str,
    out_path: Path,
    *,
    label: str,
    max_attempts: int = TTS_MAX_ATTEMPTS,
) -> float:
    """单个 chunk 重试 max_attempts 次，失败之间指数退避。

    失败时报清楚是哪个 chunk、原文是什么、试了几次、最后那次是什么异常。原来这里只留
    `str(error)`：aiohttp 的连接类异常 stringify 经常是空串，用户看到的是
    「连续 3 次合成失败：''」，既不知道是网络还是参数错，也没有 traceback 可查。
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await engine.synthesize(text, out_path)
        except _PERMANENT_ERRORS as error:
            raise _exhausted(label, text, error, attempt) from error
        except Exception as error:  # noqa: BLE001 - 网络层什么都可能抛
            if attempt >= max_attempts:
                raise _exhausted(label, text, error, attempt) from error
        await _sleep(_backoff_delay(attempt))


def _exhausted(label: str, text: str, error: Exception, attempt: int) -> TTSError:
    detail = str(error) or repr(error)
    return TTSError(
        f"{label} 合成失败（已尝试 {attempt} 次）：{text!r}\n"
        f"{type(error).__name__}: {detail}"
    )


def content_hash(text: str, fingerprint: str) -> str:
    """chunk 的内容指纹。`\\x00` 当分隔符：它不可能出现在任何一方，所以不会拼串歧义。"""
    payload = f"{fingerprint}\x00{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:CHUNK_HASH_LENGTH]


def chunk_filename(serial: int, text: str, fingerprint: str) -> str:
    """`chunk_003.1a2b3c4d.mp3`：序号在前保留人工试听时的可读性，哈希在后管身份。

    原来只有序号（`chunk_003.mp3`），而复用只检查「文件存在 + st_size > 0」。于是改写
    剧本后只要 chunk 数量和顺序碰巧一致，旧音频就被原样复用 —— 成片旁白跟它自己的字幕
    不符，零警告。哈希覆盖 (text, engine.fingerprint)，后者含 voice 与 rate。
    """
    return f"chunk_{serial:03d}.{content_hash(text, fingerprint)}.mp3"


def _find_cached_chunk(voice_dir: Path, desired: str, digest: str) -> Path | None:
    """找一个内容对得上的既有 chunk 文件。

    先看这一轮想要的那个名字，再退化成「本目录里任何一个同哈希的文件」——chunk 数量一变
    后面所有序号都会平移，但内容没变的那些没有理由重新花几十秒去合成一遍。
    """
    exact = voice_dir / desired
    if exact.is_file() and exact.stat().st_size > 0:
        return exact
    for candidate in sorted(voice_dir.glob(f"chunk_*.{digest}.mp3")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _is_pronounceable(text: str) -> bool:
    """这段文本里有没有任何「读得出声」的字符。

    唯一实现在 render/chunks.py（`is_pronounceable`）：A1 之后 split_sentences 的兜底那
    一层要用同一个判据判「这一句有没有内容可读」，两份实现随时会分叉。这里原样 re-export，
    是为了让「从 render.tts 拿 _is_pronounceable」这个既有调用面（含测试）继续成立。

    历史事故：work/saijo 的 E05 旁白用 `'…'` 当引号，chunks.split_sentences 在 `。`
    之后切开，把闭合的 `'` 留成一个独立片段（`03_script/E05.script.json` 的
    beat-3-act2 与 beat-5-act4 都有）。一旦某个 hold 正好落在这种片段上，它就会自己成为
    一个 chunk，Edge TTS 抛 NoAudioReceived（communicate.py:567），重试耗尽后整次运行
    中止 —— 一个引号搞掉一整集。切句本身的缺陷已经由 chunks.split_sentences 修掉了；
    本层这道闸留着当**防御纵深**：它保护的是「被跳过的 chunk 带的留白不能凭空消失」，
    而那条不变量的代价（此后整条时间轴前移）远大于多留几行代码。
    """
    return is_pronounceable(text)



def _plan_pronounceable(
    script: Script, warnings: list[str], *, rate: str = DEFAULT_RATE
) -> list[tuple[Beat, list[tuple[str, float]]]]:
    """切 chunk 并剔掉不可发音的那些，被剔掉的 chunk 带的留白折进前一个 chunk。

    留白是成片里真实存在的静音（render/audio.py 会把它铺出来），凭空少掉一段会让此后
    整条时间轴前移，所以只能转移、不能丢。

    `rate` 只往下传给 plan_chunks 决定 hold 落在哪个句边界上，不影响切句本身。
    `warnings` 也往下传：chunks.assign_holds 那边的两条坏 hold warning
    必须冒到这一层才看得见 —— voice 是第一个真正消费 hold.at 的阶段，也是人工改完
    03_script/*.json 之后第一个跑到的阶段（validate_script() 只在 script 阶段跑）。
    """
    staged: list[tuple[Beat, list[tuple[str, float]]]] = []
    # 指向最近一个保留下来的 chunk 所在的那个列表，跨 beat 也有效。
    last_bucket: list[tuple[str, float]] | None = None
    for beat in script.beats:
        beat_warnings: list[str] = []
        planned = plan_chunks(beat, rate=rate, warnings=beat_warnings)
        warnings.extend(f"beat {beat.id}：{message}" for message in beat_warnings)
        if not planned:
            # 只可能来自**人工编辑**：LLM 那条路上 LLMBeat.narration 是 NonBlankStr，
            # 空旁白进不来（进不来的那一刻就触发 llm.py 的 schema 修复重试）。人手清空
            # 某段旁白、只要画面不要解说是文档写明的合法编辑，所以这里降级不判错。
            # script/validate.py 的 _check_narration 会在更早的 script 阶段先报一条。
            warnings.append(f"beat {beat.id} 没有旁白文本，已跳过配音")
            continue
        kept: list[tuple[str, float]] = []
        for text, hold_after in planned:
            if _is_pronounceable(text):
                kept.append((text, hold_after))
                last_bucket = kept
                continue
            warnings.append(
                f"beat {beat.id} 有一个不含任何可发音字符的 chunk {text!r}，已跳过合成"
                "（多半是切句把闭合引号留成了独立片段）"
            )
            if hold_after <= 0:
                continue
            if last_bucket:
                previous_text, previous_hold = last_bucket[-1]
                last_bucket[-1] = (previous_text, previous_hold + hold_after)
            else:
                warnings.append(
                    f"beat {beat.id} 上面那个 chunk 带的 {hold_after:.1f} 秒留白前面没有"
                    "任何 chunk 可以挂，已丢弃"
                )
        if kept:
            staged.append((beat, kept))
        else:
            warnings.append(f"beat {beat.id} 的旁白没有任何可发音字符，已整段跳过配音")
    return staged


def _first_leaf(error: BaseException) -> BaseException:
    """从 TaskGroup 的 (Base)ExceptionGroup 里挖出第一个真正的叶子异常。

    TaskGroup 把子 task 的异常一律包成 ExceptionGroup，而 `cli.PIPELINE_ERRORS` 认的是
    TTSError **本身** —— 不解包的话「一个 chunk 的 Edge TTS 连不上」会从一行人话退化成
    一整页 traceback。取第一个叶子而不是整组：TaskGroup 在首个失败时就取消其余 task，
    所以组里通常只有一个非取消异常；真同时炸了两个也只需要报一个（都是同一次运行的
    同一类故障），多报只会淹掉重点。
    """
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            return _first_leaf(child)
    return error


async def synthesize_track(
    script: Script,
    episode: int,
    voice_dir: Path,
    engine: TTSEngine,
    *,
    reuse: bool = True,
    reporter: ProgressReporter | None = None,
    max_attempts: int = TTS_MAX_ATTEMPTS,
    previous: VoiceTrack | None = None,
    concurrency: int = DEFAULT_RENDER.tts_concurrency,
    rate: str = DEFAULT_RATE,
    ffprobe: str = FFPROBE,
) -> tuple[VoiceTrack, list[str]]:
    """合成整集旁白。chunk 独立落盘，重跑只补内容变了的那几个。

    `previous` 是上一轮的 voice.json。复用时优先取里面记下的时长，省掉每个 chunk 一次
    ffprobe 子进程 —— 文件名里的哈希已经保证内容与音色都对得上，那份时长就是同一段音频
    体检过的真实时长。

    `ffprobe` 是**复用**路径上那次时长体检要用的可执行文件（`render.ffprobe_path`）。
    刻意做成本函数的参数、而不是从 `engine` 上取：`TTSEngine` 协议就是「fingerprint +
    synthesize」两件事，ffprobe 不属于「一台 TTS 引擎是什么」——自己就能报时长的引擎
    压根用不到它，而 `getattr(engine, "ffprobe", FFPROBE)` 这种写法是在协议之外做鸭子
    类型，凡是没恰好带这个属性的引擎都会静默退回坏掉的默认值，等于把要修的 bug 重造
    一遍。复用路径体检的是**磁盘上已有**的文件（可能还是上一次运行落的），那是调用方的
    环境，跟 `voice_dir` 同一类，也跟 `rate`／`concurrency` 同一个形状。

    `concurrency` 是同时在飞的 chunk 数（`render.tts_concurrency`）。每个 chunk 的耗时
    几乎全是网络往返，所以并发几乎线性提速：115 个真实 chunk 串行 297.3 秒、并发 4
    81.9 秒、并发 24 12.8 秒（完整标定表在 `config.RenderConfig.tts_concurrency`）。
    四条不变量：

    1. **结果顺序 = 计划顺序**，跟完成顺序无关。`VoiceTrack.chunks` 的顺序决定
       render/audio.py 的 adelay 偏移与 timeline 的字幕顺序，所以结果按下标写进预分配
       的槽位，绝不 append。
    2. **同文本只合成一次**。`digest_locks` 按内容哈希给同文本的 chunk 上锁，第二个
       进临界区时 `_find_cached_chunk` 已经能 glob 到第一个落好的文件，于是走复用路径
       —— 跟串行下的行为逐字节一致。顺带把 `known_durations` 的竞态也一起关掉：这个
       dict 的每个 key 只会被「同一个哈希」的 chunk 读写，而它们全被那把锁串起来了。
       （用户报告里猜的「两个 `.part` 互相覆盖」其实不成立：文件名带序号，两个同文本
       chunk 的序号必然不同，`.part` 路径也就不同。真正的代价是一次白花的网络往返 +
       voice.json 指向两个内容相同的文件。）
    3. **失败之后不再开新活**。并发用的是「固定 N 个 worker 抢一个共享游标」，**不是**
       「给每个 chunk 建一个 task 再用 Semaphore 限流」。后者在 concurrency=1 下也会
       把 N 个 task 全建出来，首个 chunk 失败时它们已经排在同一轮事件循环里、照样会
       各发一次 Edge TTS 请求（实测：`max_attempts=1` 下第一个 chunk 就失败，engine
       仍被调了 3 次）。worker 池里游标推不动就没有新活，行为跟串行的 `for` 循环一致。
    4. **concurrency=1 与改动前逐字节等价**：一个 worker 按游标顺序取活，就是原来那个
       for 循环。
    """
    reporter = reporter or NullProgressReporter()
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    known_durations = (
        {chunk.path: chunk.duration for chunk in previous.chunks} if previous else {}
    )
    warnings: list[str] = []
    planned_by_beat = _plan_pronounceable(script, warnings, rate=rate)

    # 展平成带全局序号的作业清单。序号在这里就定下来，跟后面谁先跑完无关。
    jobs: list[tuple[Beat, int, str, float]] = [
        (beat, index, text, hold_after)
        for beat, planned in planned_by_beat
        for index, (text, hold_after) in enumerate(planned, start=1)
    ]
    total_chunks = len(jobs)

    results: list[VoiceChunk | None] = [None] * total_chunks
    digest_locks: dict[str, asyncio.Lock] = {}
    cursor = 0
    done = 0

    async def synthesize_one(position: int) -> None:
        nonlocal done
        beat, index, text, hold_after = jobs[position]
        digest = content_hash(text, engine.fingerprint)
        filename = chunk_filename(position + 1, text, engine.fingerprint)
        out_path = voice_dir / filename
        label = f"beat {beat.id} 的第 {index} 个 chunk"
        # setdefault 而不是 defaultdict：这里是纯同步代码（asyncio.Lock() 的构造不 await），
        # 所以两个 worker 之间不可能插进来各建一把锁。
        lock = digest_locks.setdefault(digest, asyncio.Lock())
        async with lock:
            cached = _find_cached_chunk(voice_dir, filename, digest) if reuse else None
            if cached is not None:
                filename = cached.name
                duration = known_durations.get(filename)
                if duration is None:
                    duration = await asyncio.to_thread(
                        probe_duration, cached, ffprobe=ffprobe
                    )
            else:
                duration = await synthesize_with_retry(
                    engine, text, out_path, label=label, max_attempts=max_attempts
                )
            # 同一段文字在剧本里出现两次时，第二个 chunk 会命中第一个的文件；记下来
            # 就连那一次 ffprobe 也省了。锁内写、锁内读，所以并发下也没有竞态。
            known_durations[filename] = duration
        results[position] = VoiceChunk(
            beat_id=beat.id,
            index=index,
            text=text,
            path=filename,
            duration=duration,
            hold_after=hold_after,
        )
        # 完成计数器，不是循环下标 —— 并发下下标会乱序。label 是「刚刚完成的那一句」。
        done += 1
        reporter.substep("voice", done, total_chunks, text[:20])

    async def worker() -> None:
        nonlocal cursor
        while True:
            if cursor >= total_chunks:
                return
            # 取号与自增之间没有 await，所以两个 worker 不可能拿到同一个号。
            position = cursor
            cursor += 1
            await synthesize_one(position)

    # total_chunks=0 时 worker 数也是 0，TaskGroup 空跑一轮就退出 —— 不需要额外的空判。
    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(min(max(1, concurrency), total_chunks)):
                group.create_task(worker())
    except BaseExceptionGroup as error:
        leaf = _first_leaf(error)
        # `from leaf.__cause__` 而不是 `from error.__cause__`：TaskGroup 抛的组
        # `__cause__` 是 None，写成后者等于 `raise leaf from None`，会把
        # synthesize_with_retry 挂上去的那个原始网络异常从 __cause__ 里抹掉
        # （aiohttp 的连接类异常 stringify 常常是空串，这条链是唯一的线索）。
        # 而 `from` 的另一半作用（suppress_context）是想要的：ExceptionGroup
        # 自己是噪音，不该再作为 "During handling of the above exception" 印一遍。
        raise leaf from leaf.__cause__

    # 全部 chunk 都成功时 results 里不可能还有 None（每个 position 恰好被写一次），
    # 但类型上它是 VoiceChunk | None，所以这里显式过滤给类型检查器看。
    chunks = [chunk for chunk in results if chunk is not None]
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=episode, chunks=chunks, total_seconds=total), warnings
