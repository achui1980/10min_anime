# Anchor 走廊修复与泛化 设计

## 背景 / 动机

用户贴出的一次真实运行日志（`work/saijo/` E11）里出现了六类 `[warn]`，用户的问题（原话）：

> 我看到有一些这些warn是做什么的,它有什么影响

排查发现这六条 warning 全部集中在 `src/tenmin/script/validate.py`（5 条）和 `src/tenmin/render/subtitles.py`（1 条），其中最要命的两条是：

```
[warn] 阶段五：练习赛与赌注：clip 起点 1380.0 与 anchor 行时间 1141.6 差了 238.4 秒，超过覆写上限 60 秒，没有覆写（anchor_lines 或 clip 时间戳之一是系统性错的，请人工核对）
[warn] 阶段五：练习赛与赌注：clip 1380.0-1420.0 的 anchor 行有 2/2 条落在时间窗外，旁白讲的内容缺画面
```

对 `work/saijo`、`work/akujo2`、`work/saijo2`（实际是《我是不才恶女》E01，`slug` 命名有误但不在本次修复范围）三个项目、13 份 `03_script/*.script.json` 做的实测分析（read-only，脚本在 `/tmp`，未改仓库）确认了根因：

1. **LLM 编造 clip 时间戳是真实存在且有规律的现象**——`clip.start` 健康时几乎精确复制某个 anchor 行的 `start`（`drift < 0.05s`），出问题时则是整数/整 5/整 10 的"编造指纹"，且在同一 beat 内呈**单调递增**、首尾相接铺出一条连续时间带的模式（LLM 是按"这个 beat 的旁白要配多少秒画面"算出一个时间带，而不是去 anchor 真实位置取素材）。
2. 现有的硬编码上限 `ANCHOR_OVERWRITE_MAX_SECONDS = 60.0`（`validate.py:56`）恰好把**唯一可以确定性修复的那批**（drift 84~238 秒但 anchor 行内容与画面完全自洽）挡在了"没有覆写，人工核对"的死角——这批 clip 在 saijo2 E01 上占了 8/9 个错位 clip，导致成片 **31%（70.9s/232.1s）的画面配错戏**，且完全没有自动修复路径。
3. 该常量的注释自己承认"它是从推理来的，不是从数据来的"（10 个健康样本一次都不触发，没有"合法覆写"样本可标定）——**它不是一个可信的物理边界，只是一个猜的数字**，换一部番、换一个片长就可能失效或误伤。
4. 但"无条件信 anchor 重建"也不成立：saijo2 E01 里有一个 clip（`269.0-346.0`，drift=-320.8）的 anchor 行内容与 visual 描述完全自洽，却指向**阶段三**（地牢真相剧情），被放进了**阶段一**的 beat——如果无条件按 anchor 重建，会把后面剧情的画面提前塞进前面，造成剧透。
5. 因此需要一个**不引入新的绝对秒数阈值、只做集内相对判断**的判别器，用来区分"anchor 指向本节点、可以重建"和"anchor 指向别的节点、应该丢弃"。

另外还发现两个连带 bug：
- **Bug A**：`track.op_range is None`（OP 片头检测失败）时，`_reject_reason` 里的片头重叠检查（`validate.py:139-144`）直接失效（`_credits_overlap_ratio` 对 `window=None` 返回 0.0），saijo2 E01 的 hook clip 里因此有 12.5 秒放的是纯 OP staff roll，而且**完全静默**，日志里看不到任何信号。
- **Bug C**：ED 重叠判据是"占 clip 时长 ≥ 50% 才丢整条"（`CREDITS_OVERLAP_MAX_RATIO = 0.5`），saijo2 E01 最后一个 clip 压到 ED 只有 16%，没有被丢，成片结尾会闪一下片尾曲画面。

其余四条 warning（Warning 4/5/6，对应"留白金句找不到原声""anchor 行落在窗外""字幕折成 3 行"）经排查确认：Warning 4/5 是 Warning 3（60 秒上限拒绝覆写）不修的连带症状，一旦按 anchor 正确重建窗口，二者在定义上基本消失；Warning 6 是纯粹的旁白文案长度问题，与时间戳对齐无关，本次顺带做一个低成本改进（把 warning 喂回 LLM 重试），不做拆句/放宽行数等结构性改动。

## 已确认的设计决策（Q&A）

1. **覆写力度**：不设绝对秒数上限，改成**走廊检验**——超容差的 clip 只要 anchor 落在"相邻已知健康节点围成的走廊"内就重建，否则丢弃（而不是保留一个更宽的绝对上限如 180 秒，也不是靠"clip.start 是否整 5/整 10"这种指纹判断）。理由：走廊是集内相对量，不随片长/番剧/语言变化，且已用 13 份真实语料验证过零误伤。
2. **离群 clip 处置**：判定为"anchor 指向别的节点"时，直接**丢弃该条 clip + 发 warning**（复用项目现有的"丢单条、保其余"降级语义），而不是仍按 anchor 重建（会剧透，已有反例证实），也不是让整份剧本抛错重跑（代价过大，且该 beat 通常还有其他健康 clip 补位）。
3. **体质检测**：在 `repair_script` 入口新增一个"整份剧本健康率"检测，健康率低于阈值直接判定为"整集行号/时间体系不同源"（如 saijo E02），走现有的 `ScriptValidationError` 重试路径，不逐条修复。阈值 `0.25` 是新增的模块级常量（不进 `config.py`，理由与现有 `ANCHOR_OVERWRITE_MAX_SECONDS`/`CREDITS_OVERLAP_MAX_RATIO` 一致：这是"数据坏没坏"的合法性边界，不是创作旋钮）。
4. **hold 金句定位**：只提升 `_quote_matches` 的匹配算法（加 `difflib.SequenceMatcher`），**不做自动扩窗**去够金句——这会破坏项目一贯的"只报不改"风格，且扩窗本身需要一个新的绝对秒数阈值，与决策 1 的方向矛盾。
5. **Warning 6（字幕折行）**：本次只做"把 `check_script`/`check_cue_legibility` 产出的 warning 一并喂回校验重试的 prompt"这一个改动，让模型有机会自己把旁白写短。**不做**自动拆句、`chunks.py` 二次切分、放宽 `subtitle_max_lines` 到 3——这些都是需要新数据支撑或改变产物契约的独立课题。
6. **credits 检测算法（Bug A）**：本次不修 `ingest/credits.py` 里的 OP 检测逻辑本身，只做到"`op_range`/`ed_range` 为 `None` 时不再静默、发 warning"——把隐藏 bug 变成可见 bug，让人工介入有信号可依。真正的 staff roll 识别算法改进单独立项。
7. **ED 重叠（Bug C）**：改成"clip 只要和 ED 窗口有重叠（不到 50% 丢弃阈值）就把重叠部分钳掉"，而不是提高或降低 `CREDITS_OVERLAP_MAX_RATIO` 这个已有实测支撑的常量。

## 设计详情

### Section 1：体质检测闸门（新增，对应 Bug：saijo E02 式整集错位）

**问题**：`_apply_anchor` 是逐 clip 独立判断的，遇到"整集 anchor 行号体系跟 clip 时间戳不同源"（比如 script.json 是针对旧版 `dialogue.json` 生成的 stale 产物）时，几乎每个 clip 都会被判定为"超容差"，逐条按后面 Section 2 的走廊规则处理只会得到一堆随机的重建/丢弃决定，修不出一份连贯的剧本。这种情况必须整份重跑，而不是逐条修补。

**判别依据（实测数据）**：对 13 份 `script.json` 计算"健康率 = anchor drift 落在 `cfg.anchor_tolerance_seconds` 容差内的 clip 占比"：

| 语料 | 健康率 |
|---|---|
| saijo E02（整集不同源，已知坏） | **4%**（1/23） |
| akujo2 E02（健康对照） | 22%（4/18） |
| saijo2 E01（局部错位，可修） | 52%（13/25） |
| saijo E11（局部错位，可修） | 64%（16/25） |
| 其余 9 份健康剧本 | 均 ≥ 16% |

坏样本（4%）与最差的"仍可逐条修"样本（52%）之间有巨大间距，取中点附近的 **25%** 作为分界，两侧都留有余量。

**实现**：

```python
# validate.py 模块级常量区（原 ANCHOR_OVERWRITE_MAX_SECONDS 所在位置附近）
HEALTHY_ANCHOR_MIN_RATIO = 0.25
# 整份剧本里，anchor drift 落在容差内的 clip 占比低于这个值，
# 说明不是"个别 clip 编造"，是"整集行号/时间体系跟 clip 时间戳不同源"，
# 逐条修不可靠，必须整份重跑。0.25 是从 13 份真实语料的健康率分布里取的，
# 坏样本 4% 与最差可修样本 52% 之间留了两倍余量。

def _healthy_anchor_ratio(
    script: Script, indexes: dict[int, AnchorIndex], cfg: ValidateConfig
) -> float:
    """全剧本范围内，anchor drift 落在容差内的 clip 占比。
    分母只统计"能找到 anchor 行"的 clip；anchor_lines 引用不到任何行的
    clip 不计入统计（那是另一个问题，不该稀释这个比率）。
    """
    total = 0
    healthy = 0
    for beat in script.beats:
        for clip in beat.clips:
            index = indexes.get(clip.episode)
            if index is None:
                continue
            matches = index.matches(clip.anchor_lines)
            if not matches:
                continue
            anchor_start = min(line.start for line in matches)
            total += 1
            if abs(anchor_start - clip.start) <= cfg.anchor_tolerance_seconds:
                healthy += 1
    return healthy / total if total else 1.0
```

调用位置：`repair_script` 里 `indexes = _anchor_indexes(tracks)`（现 552 行）之后、逐 beat 循环之前：

```python
    ratio = _healthy_anchor_ratio(script, indexes, cfg)
    if ratio < HEALTHY_ANCHOR_MIN_RATIO:
        raise ScriptValidationError(
            f"整份剧本只有 {ratio:.0%} 的 clip 时间戳与 anchor 行吻合"
            f"（阈值 {HEALTHY_ANCHOR_MIN_RATIO:.0%}），疑似整集行号/时间体系不同源"
            f"（比如剧本是针对旧版字幕生成的 stale 产物），逐条修复不可靠，剧本不可用，重试",
            script=script,
        )
```

这条检查在原有的"节点数不足"检查（544 行）之后、逐 beat 修复循环之前，走的是同一个 `ScriptValidationError` 重试路径，不需要新的错误类型或新的 pipeline 分支。

### Section 2：走廊检验取代 60 秒硬上限（对应决策 1、2）

**删除** `ANCHOR_OVERWRITE_MAX_SECONDS = 60.0`（`validate.py:56`）及其在 `_apply_anchor` 里"超过就不覆写"的分支。

**新增判别依据（实测数据，已在 13 份语料上验证）**：排除 `role in ("hook", "outro")` 的 beat 后，中间"阶段一..阶段 N"这些 `role in ("act", "climax")` 的 beat，其健康 clip（`|drift| <= cfg.anchor_tolerance_seconds`）的 anchor 起点中位数（"领地"），在全部 13 份剧本上**严格单调递增**——这是叙事结构（按剧情顺序讲解）决定的不变量，与番剧、片长、模型无关，可以作为不引入新绝对阈值的判别基础。

对每个超容差的 clip，取它所在 beat 左右最近的"有领地"的 act/climax beat 的领地边界，围成一条"走廊"：anchor 落在走廊内 → 判定"anchor 指向本节点，只是被编造的时间戳骗了" → 重建；anchor 落在走廊外 → 判定"anchor 大概率指向别的节点" → 丢弃这条 clip（不覆写、不保留），保留该 beat 的其余 clip。`role in ("hook", "outro")` 的 beat 不参与领地计算，其内部的超容差 clip 直接免检重建——这与现有的 `_check_timeline_order`（A1，`validate.py:288-324`，`beat.role in ("hook","outro")` 同样免检，`validate.py:310`）用的是同一套 role 语义，不新造 Hook/收尾 判断方式。

**实测结果**（13 份 script.json）：saijo E02 被 Section 1 的体质检测拦下（不进入这一步）；其余 12 份里，共判定 **14 处重建、13 处丢弃**（其中 saijo2 E01 的 `269.0-346.0` 那个反例正确落在"丢弃"一侧）；**9 份完全健康的剧本（E01, E03–E10, akujo2）零改动**——回归安全性有实测保证。

```python
def _beat_territories(
    beats: list[Beat], indexes: dict[int, AnchorIndex], cfg: ValidateConfig
) -> list[tuple[float, float] | None]:
    """给每个 role in ("act","climax") 的 beat 算一个"领地"：
    健康 clip（|drift| <= tolerance）对应 anchor 行的 [最早起点, 最晚终点]。
    Hook/收尾 beat 以及没有任何健康 clip 的 beat 记为 None（不参与围走廊）。
    """
    territories: list[tuple[float, float] | None] = []
    for beat in beats:
        if beat.role not in ("act", "climax"):
            territories.append(None)
            continue
        starts: list[float] = []
        ends: list[float] = []
        for clip in beat.clips:
            index = indexes.get(clip.episode)
            if index is None:
                continue
            matches = index.matches(clip.anchor_lines)
            if not matches:
                continue
            anchor_start = min(line.start for line in matches)
            if abs(anchor_start - clip.start) <= cfg.anchor_tolerance_seconds:
                starts.append(anchor_start)
                ends.append(max(line.end for line in matches))
        territories.append((min(starts), max(ends)) if starts else None)
    return territories


def _corridor_for(
    position: int, territories: list[tuple[float, float] | None], track_duration: float
) -> tuple[float, float]:
    """给定 beat 下标，找左右最近的"有领地"的 act/climax beat，围出走廊。
    找不到就钳到 [0, track_duration]。
    """
    lower = 0.0
    for i in range(position - 1, -1, -1):
        if territories[i] is not None:
            lower = territories[i][1]
            break
    upper = track_duration
    for i in range(position + 1, len(territories)):
        if territories[i] is not None:
            upper = territories[i][0]
            break
    return (lower, upper)
```

`repair_script` 里逐 beat/逐 clip 循环（现 557-593 行）的改动点：在 `_apply_anchor` 之前，先按 `beat.role` 分流；`role in ("hook","outro")` 的直接调用 `_apply_anchor`（见 Section 3，改成无条件重建）；其余的先算走廊，落在走廊内才调用 `_apply_anchor`，落在走廊外则直接生成丢弃 warning 并 `continue`（不进入后面的 `_reject_reason` 检查，也不 `kept.append`）：

```python
    territories = _beat_territories(repaired.beats, indexes, cfg)
    for beat_position, beat in enumerate(repaired.beats):
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：引用了不存在的集数 {clip.episode}")
                continue
            index = indexes[clip.episode]
            matches = index.matches(clip.anchor_lines)
            anchor_start = min((line.start for line in matches), default=None)
            drift = abs(anchor_start - clip.start) if anchor_start is not None else 0.0

            if anchor_start is not None and drift > cfg.anchor_tolerance_seconds:
                if beat.role not in ("hook", "outro"):
                    lower, upper = _corridor_for(beat_position, territories, track.duration)
                    if not (lower <= anchor_start <= upper):
                        warnings.append(
                            f"{beat.label}：clip 起点 {clip.start:.1f} 与 anchor 行时间 "
                            f"{anchor_start:.1f} 差了 {drift:.1f} 秒，且 anchor 落在本节点走廊 "
                            f"[{lower:.1f}, {upper:.1f}] 之外（更像是指向别的节点），丢弃这条 clip"
                        )
                        continue
                warnings.extend(_apply_anchor(clip, track, index, beat.label, cfg))
            # ...后续 _reject_reason / is_silent_highlight 回填逻辑不变
```

### Section 3：重建规则改成按 anchor 行簇定窗（对应 Warning 3、5 的连带消失）

**问题**：现有 `_apply_anchor` 重建时只用 `index.earliest_start`（`validate.py:183-188`，只取 `min(starts)`）覆写 `clip.start`，`clip.end` 靠"保留原 `clip.duration`"算出来——但原时长是 LLM 编造出来的坏值，重建后的窗口经常还是不够覆盖全部 anchor 行，就是 Warning 5（"anchor 行落在窗外，旁白缺画面"）的来源。

**新规则**：重建时同时用 anchor 行簇的起止两端定窗，并保证不小于 clip 原时长：`start = min(anchor 行 start)`，`end = max(max(anchor 行 end), start + 原时长)`，最终钳到 `track.duration`。

```python
def _apply_anchor(
    clip: Clip, track: DialogueTrack, index: AnchorIndex, label: str, cfg: ValidateConfig
) -> list[str]:
    """把 clip 的时间窗改成覆盖它引用的全部 anchor 行（保底不小于原时长）。
    调用者负责判断"要不要调用"（容差内不调、走廊外不调，见 Section 2）；
    这个函数本身只负责"调用了就一定重建"。
    """
    matches = index.matches(clip.anchor_lines)
    if not matches:
        return []
    anchor_start = min(line.start for line in matches)
    anchor_end = max(line.end for line in matches)
    duration = clip.duration
    new_start = anchor_start
    new_end = min(max(anchor_end, anchor_start + duration), track.duration)
    clip.start, clip.end = new_start, new_end
    return [
        f"{label}：clip 起点 与 anchor 行时间 {new_start:.1f} 偏差超过 "
        f"{cfg.anchor_tolerance_seconds:.0f} 秒，以字幕时间为准"
    ]
```

由于新窗口天然覆盖全部 anchor 行，`_check_anchor_coverage`（Warning 5，`validate.py:264-284`）对刚被重建过的 clip 在定义上不会再报——这条 warning 只会在"clip 完全没被 Section 2 判定为超容差"（也就是本来就健康）却仍有 anchor 落在窗外的情况下出现，属于独立于本次修复的边缘情况，保留不动。

### Section 4：credits 不再静默 + ED 部分重叠钳位（对应决策 6、7，Bug A / Bug C）

**Bug A 的可见化**：`build_track`（`ingest/normalize.py`）产出的 `DialogueTrack.op_range`/`ed_range` 为 `None` 时，现在没有任何信号。在 `repair_script` 里拿到 `track` 后立即检查并发 warning（每集每种缺失只报一次，用 `reported_missing_credits_op`/`reported_missing_credits_ed` 两个 `set[int]` 分别去重，和现有 555 行的 `reported_missing`（SignalReport 缺失去重）用同一套模式）：

```python
    if track.op_range is None and clip.episode not in reported_missing_credits_op:
        warnings.append(f"第 {clip.episode} 集片头曲区间检测失败（op_range 为空），OP 重叠检查对这一集形同虚设")
        reported_missing_credits_op.add(clip.episode)
    if track.ed_range is None and clip.episode not in reported_missing_credits_ed:
        warnings.append(f"第 {clip.episode} 集片尾曲区间检测失败（ed_range 为空），ED 重叠检查对这一集形同虚设")
        reported_missing_credits_ed.add(clip.episode)
```

真正修复 OP 检测算法（让 `op_range` 不再是 `None`）属于 `ingest/credits.py` 的独立课题，本次不做。

**Bug C 的钳位**：在 `_reject_reason`（`validate.py:120-145`）判断"重叠 ≥ 50% 就整条丢弃"之前，新增一步钳位：只要 clip 与 OP/ED 窗口有重叠但没到丢弃阈值，就把重叠部分切掉，而不是任由它露出片头/片尾画面：

```python
def _clamp_credits_overlap(clip: Clip, track: DialogueTrack) -> None:
    """clip 与 OP/ED 窗口部分重叠（未达 CREDITS_OVERLAP_MAX_RATIO 丢弃线）时，
    把压在片头/片尾里的那一段切掉，而不是放着不管。就地改 clip。
    """
    if track.op_range is not None and clip.start < track.op_range[1]:
        clip.start = max(clip.start, track.op_range[1])
    if track.ed_range is not None and clip.end > track.ed_range[0]:
        clip.end = min(clip.end, track.ed_range[0])
```

调用位置：`repair_script` 里 `_apply_anchor` 之后、`_reject_reason` 之前（保持"先用最终时间戳跑判据"的既有顺序，`validate.py:536-538` 的 docstring 已经这么写）。`CREDITS_OVERLAP_MAX_RATIO = 0.5`（`validate.py:69`）不变——它判断的是"钳完之后还压了一半以上，说明这条 clip 本质上就是在讲 OP/ED，整条丢弃"，钳位与丢弃阈值是互补关系，不是替代关系。

### Section 5：hold 定位率提升（对应决策 4）

`_quote_matches`（`validate.py:205`）目前只做归一化子串包含匹配，实测 61 个真实 hold 定位率 75.4%，约 25% 静默跳过（`_check_hold_quotes` 里 `if not matches: continue`，`validate.py:247-248`）。加一层基于标准库 `difflib.SequenceMatcher` 的相似度兜底：先跑现有的子串匹配，找不到再用 `SequenceMatcher` 算相似度，取相似度最高且超过阈值的若干行：

```python
import difflib

_QUOTE_SIMILARITY_MIN_RATIO = 0.6  # 与现有 QUOTE_FRAGMENT_MIN_RATIO 同量级，模块级常量

def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
    target = _normalize_quote(quote)
    minimum = max(QUOTE_FRAGMENT_MIN_CHARS, QUOTE_FRAGMENT_MIN_RATIO * len(target))
    exact: list[DialogueLine] = []
    best_fuzzy: tuple[float, DialogueLine] | None = None
    for episode in episodes:
        track = tracks.get(episode)
        if track is None:
            continue
        for line in track.lines:
            normalized = _normalize_quote(line.text)
            if target in normalized or (len(normalized) >= minimum and normalized in target):
                exact.append(line)
                continue
            ratio = difflib.SequenceMatcher(None, target, normalized).ratio()
            if ratio >= _QUOTE_SIMILARITY_MIN_RATIO and (best_fuzzy is None or ratio > best_fuzzy[0]):
                best_fuzzy = (ratio, line)
    if exact:
        return exact
    return [best_fuzzy[1]] if best_fuzzy else []
```

`SequenceMatcher` 只在**精确/子串匹配全部失败**时才启用，不改变现有 75.4% 的判定结果，只补那 25%。不做自动扩窗——找到的行如果仍不在任何 clip 窗口内，`_check_hold_quotes` 照常报 Warning 4，行为与现在一致，只是"找不到"的比例会下降。

### Section 6：warning 喂回重试 prompt（对应决策 5）

`single.py:330-340` 现在校验失败重试时只把 `str(error)`（`ScriptValidationError` 的那一条消息）塞进下一轮 prompt，`check_script()` 产出的整堆 warning（包括 Warning 6 字幕折行）从来没进过 prompt，模型没有机会看到、也没有机会自我修正。改成把上一轮 `check_script` 的 warning 列表一并附上：

```python
            warnings.append(f"第 {attempt} 轮剧本校验失败，重试：{error}")
            issues = "\n".join(f"- {w}" for w in stage_warnings) if stage_warnings else ""
            extra = f"\n\n此外还有以下问题（尽量一并修掉）：\n{issues}" if issues else ""
            prompt = (
                _followup_prompt(
                    followup_base,
                    error.script,
                    "上一轮的问题",
                    f"{error}{extra}\n请在上一版基础上修掉这个问题，"
                    f"确保每个节点至少有一个有效 clip，且所有时间戳都落在正片范围内。",
                )
                if error.script is not None
                else f"{followup_base}\n\n## 上一轮的问题\n\n{error}{extra}\n请重新输出。"
            )
```

`stage_warnings` 需要在 `draft()` 里跟随 `error.script` 一起带出来（目前 `check_script`/`repair_script` 的 warning 在 `ScriptValidationError` 抛出之前就已经生成，只是没被夹带传递）——具体接线在实施计划阶段确认 `draft()` 内部的变量传递路径。

## 测试计划

- 改 `tests/test_validate.py`：
  - `test_anchor_overwrite_beyond_the_cap_is_refused_with_a_warning`（497）→ 改成"anchor 落在走廊外，丢弃"场景。
  - `test_anchor_overwrite_within_the_cap_still_happens`（510）→ 改成"anchor 落在走廊内，按 anchor 行簇重建"场景。
  - `test_anchor_overwrite_cap_constant`（517-518）→ 删除（常量已不存在）。
- 新增单元测试：走廊内重建、走廊外丢弃、Hook 免检重建、收尾免检重建、体质检测抛错（健康率 < 25%）、健康剧本零改动、`op_range`/`ed_range` 为 `None` 时发 warning（每集只报一次）、ED 部分重叠钳位、`_quote_matches` 的 `SequenceMatcher` 兜底命中。
- 语料回归测试（可挂 `-m generalize`，需要真实 `work/` 数据，不进默认 CI 路径）：对 13 份 `script.json` 跑一遍新逻辑，断言：9 份健康剧本（E01, E03–E10, akujo2）零改动；saijo2 E01 得到 8 处重建 + 1 处丢弃；saijo E02 被体质检测拦下抛错。

## 不在本次范围

- OP credits 检测算法修复（`ingest/credits.py`，让 `op_range` 不再是 `None`）——本次只做到发 warning。
- edge-tts `SentenceBoundary` 元数据用于拿真实句边界（`timeline.py:66-69` 记录的线索）——可以消除 Section 6 之外、句内时间估算误差（p50 0.365 / p90 0.817 / max 1.615 秒）这个独立问题，需要改变 voice 阶段的产物契约，单独立项。
- `work/saijo2/project.yaml` 里 `show: saijo2` 与实际内容（我是不才恶女 E01）不符的命名错误——这是用户本地数据的问题，不是代码 bug。
- Warning 6 的结构性修法（`chunks.py` 按逗号二次切分、放宽 `subtitle_max_lines` 到 3）——本次只做 Section 6 的"喂回重试"，效果如不理想再单独立项。
