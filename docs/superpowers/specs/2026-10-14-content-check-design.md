# 最终成片内容检查设计

日期：2026-10-14

状态：聊天方案已确认；本文待用户审阅后进入实现计划。

## 1. 目标与已确认决策

把动漫分级 POC 整理为独立的本地内容检查流程，同时提供项目管线和独立 CLI 两个入口。

- 项目入口只检查最终成片，不扫描原始番剧。
- 每集 render 之后执行 check，报告时间对应最终成片的播放时间。
- 检出候选只生成报告、提示数量，继续后续集数。
- 检查执行失败时，项目管线明确警告、保留成片、继续后续集数。
- 独立命令可检查任意视频，不要求 project.yaml，不调用 LLM 或 TTS。
- 全程本地推理，首次安装依赖、下载模型需要联网；缓存后可离线。

本期不做自动删除、打码、剪辑调整、人工审核状态管理或原片与成片时间轴映射。

## 2. 依据与模型边界

POC：`docs/poc/anime_rating_poc.py`，实测记录见 `docs/poc/README.md`。

使用 `deepghs/anime_dbrating`，固定仓库 revision：
`7af21db648acdeb74f5c334abda9dd7403407b3c`。
首版仅提供已经实测的 `mobilenetv3_large_100_v0_ls0.2`，约 17MB，CPUExecutionProvider。
大模型后续独立评估，本期不把未试跑的模型变成配置选项。

四类为 general、sensitive、questionable、explicit；展示原始英文评级与中文解释。
输出是模型图片评级分数，不是校准后的色情概率，也不是严重程度百分制。

真实素材中泳装腰臀特写可能被判 explicit，男性普通脸部特写也可能被列为候选。
候选表示值得人工查看，不表示已确认色情或裸露。抽帧不能保证发现短于采样间隔的镜头。

## 3. 接入方案与模块边界

采用共享检查服务、两个薄入口：

1. `src/tenmin/content_check/`：抽帧、模型推理、缓存、候选整理、报告生成。
2. `pipeline.py`：注册 check 阶段与按集产物路径，调用共享服务。
3. `cli.py`：新增 check-content；现有 run 支持 check 阶段并打印报告位置。
4. `config.py`：ContentCheckConfig 与默认配置实例，集中管理用户可调参数。
5. `config_slices.py`：增加 check 切片，确保检查配置不影响其他阶段。

共享服务接收视频路径、输出目录、检查配置、ffmpeg/ffprobe 路径及 force/offline 选项；
返回结构化结果，包含完成状态、候选数和报告路径。核心模块不读取项目配置文件，
不 import CLI 或管线，不自行决定是否终止批处理。

相比把逻辑放进 render，独立阶段可以重查现有成片，不必重新渲染。
相比另建一份 CLI 实现，共享服务能保持两种入口的分数、缓存及报告语义一致。

## 4. 配置与触发规则

```yaml
content_check:
  enabled: false
  sample_interval: 1.0
  sensitive_threshold: 0.5
  adult_threshold: 0.5
```

- enabled 默认 false，保持存量项目行为；用户设置 true 后随正常制作流程执行。
- sample_interval 为有限正秒数；两个阈值均为有限数，范围 [0, 1]。
- 显式 `--only check` 视为主动请求，忽略 enabled 开关。
- 显式 `--from check` 同样主动执行；`--only render` 只渲染，不隐式追加 check。
- 普通全流程及 `--from render` 在 enabled=true 时包含 check。
- 普通全流程 enabled=false 时明确显示检查未启用。
- 显式检查缺少最终成片时提示先完成 render，不自行启动上游制作。

阶段列表扩展为：

```text
ingest → translate → signals → script → docgen → voice → timeline → audio → render → check
```

check 单独执行时不构造 LLM provider 或 TTS engine。保持现有按集纵向编排，
每集检查结束后进入下一集。

## 5. CLI

以下是实现后的目标接口，当前尚未提供：

```bash
# 启用项目开关后，随制作流程检查最终成片
uv run tenmin run saijo --episode 2

# 单独检查项目现有成片，不需要 API key
uv run tenmin run saijo --episode 2 --only check

# 强制重新抽帧、推理
uv run tenmin run saijo --episode 2 --only check --force

# 独立视频，默认输出到视频旁的 <stem>.content-check/，也可指定目录
uv run tenmin check-content 视频.mp4 --output 检查报告目录

# 参数覆盖与离线模型缓存
uv run tenmin check-content 视频.mp4 --interval 1 \
  --sensitive-threshold 0.5 --adult-threshold 0.5 --offline
```

独立命令首版只收一个视频文件，参数包括 output、interval、两个阈值、force、offline。
不增加目录批量扫描或模型切换。配置默认值与项目入口共用。
offline 只控制模型读取本地缓存；uv 依赖解析是否联网由 uv 自己的离线选项控制。

候选数大于零仍退出 0；独立命令无法完成检查退出 1；CLI 参数格式错误沿用 Typer 语义。
项目管线发生可预期的检查错误时输出警告并继续，结束时汇总检查失败的集数。
不捕获 KeyboardInterrupt、取消请求或未知编程错误来伪装完成。

## 6. 检测流程

1. 校验视频及参数，确认依赖和模型可用。
2. 检查缓存身份；可复用则跳过抽帧和推理。
3. FFmpeg 每约 sample_interval 秒选取一帧；首帧也采样。
4. 用 showinfo 的真实 PTS 记录成片播放时间，不用帧序号乘间隔推算。
5. 保存宽 640、保持显示比例的 JPEG 截图，再按官方输入预处理推理。
6. 生成所有采样帧的四类分数、top-1 评级，整理候选采样点范围。
7. 写入 JSON 与 HTML，打印候选数和报告位置。

预处理按官方 imgutils：RGB、双线性缩放、CHW、归一化到 [-1, 1]；
空间维度为动态时用 384×384。输出已经是分数，不重复 softmax。
初版 CPU 单会话、4 线程，逐帧推理，不引入并行下载和多模型投票。
抽帧处理视频显示方向和非方形像素，确保截图时间与外观对应正常播放。

每帧候选规则：

```text
sensitive >= sensitive_threshold
或 questionable + explicit >= adult_threshold
```

排序值为 `max(sensitive, questionable + explicit)`，仅用于查看优先级，不命名为严重程度。
按时间顺序将相邻候选采样点间隔不超过 1.5 × sample_interval 的点归组，
展示首尾采样时间与帧数；不把采样点范围宣称为精确片段边界。

## 7. 产物、缓存与失败一致性

项目每集输出目录：`work/<slug>/08_check/E{NN}/`，由 Paths 统一提供，内含：

```text
results.json    # 全部帧原始分数、视频/模型/采样身份、当前阈值下的候选及耗时
report.html     # 自包含页面逻辑，图片使用相对路径
frames/        # 所有采样截图，支持查看漏报
```

JSON 包含 schema_version、输入视频绝对路径、size、mtime_ns、模型 repo/revision/文件摘要、
预处理版本、采样间隔、执行 provider、耗时以及逐帧结果。记录阈值和候选汇总，
但不让阈值成为原始推理缓存的身份。

缓存身份包括视频 resolve 路径/size/mtime_ns、固定模型版本及权重摘要、
采样与预处理版本。不是完整视频哈希，无法识别内容变化但大小和 mtime 被刻意保留的文件；
这种情况用 --force 重跑。

- 身份相同、结果完整且所有引用截图存在：复用分数。
- 仅阈值变化：重算候选与报告，不抽帧、不推理。
- 报告缺失、原始结果有效：重建报告。
- 截图缺失、JSON 损坏、身份不同或 force：重新检测。
- check 的配置切片变化只触发检查服务，由服务判断需要重算报告还是重新推理。
- 即使管线 render 因新鲜度跳过，仍独立判断 check 是否需要执行。
- 缓存命中仍输出当前候选数与报告路径，避免静默丢失既有提示。

使用隔离的临时工作目录抽帧与推理，避免旧截图混入。完成后替换本模块管理的产物，
以最后原子写入的 results.json 作为完成标记；在替换前先使旧完成标记失效。
中断或写入失败的半成品不满足缓存有效条件，下次重跑。
不要递归删除用户指定目录中的其他文件；非本工具拥有的非空输出目录应报错。

失败时明确显示“本次检查失败／报告未更新”，旧报告不得作为当前视频检查成功的证据。
零候选与未检查、检查失败是三个不同状态；零采样帧属于失败。

## 8. 报告呈现

- 顶部显示输入视频、模型版本、采样间隔、采样数、耗时与候选数。
- 每张卡片显示时间、截图、原始四类分数和模型 top-1 评级。
- 默认只看候选，支持全帧查看、按时间或筛选分数排序、分页加载。
- 页面内调整阈值立即过滤，无需联网或重新推理；页面调整不回写项目配置或 JSON。
- 显示候选采样点范围，并注明不是精确边界。
- 不把评级包装为确定的色情判断，不显示百分制严重程度。
- HTML 内嵌 JSON 安全转义，视频名等内容以文本插入。

## 9. 依赖与兼容性

新增可选 extra `content-check`，包含 ONNX Runtime、NumPy、Pillow、huggingface-hub；
只有真正进入检查服务才延迟导入。安装引导为 `uv sync --extra content-check`。
不启用时不要求检查依赖、模型、API key，不访问模型下载服务。
模型 revision 固定，利用 Hugging Face 缓存；已缓存后默认也应避免不必要的远端查询。
缺依赖、缺离线模型、下载失败均给出可操作的中文错误。

依赖版本和 Python 版本标记在实施时以实际解析及运行结果确定。
POC 的 Python <3.14 限制不能直接变成主项目全局限制；如某 Python 版本没有可用 ONNX wheel，
应限制 extra 的适用范围并明确提示，不影响未启用检查的项目运行。

更新 README 与 AGENTS.md 的阶段、配置数量及新命令说明，保留现有产物路径与历史设计快照。

## 10. 验收

1. 配置默认关闭；开启后每集 render 完成即检查，候选不会阻止后续集。
2. 显式 --only check 在开关关闭时仍可检查已有成片，不构造 LLM/TTS。
3. 独立命令不需要项目；同一视频和配置，两种入口的帧分数与候选判定一致。
4. 单元测试覆盖阈值边界、候选归组、原始分数保留、PTS 与截图对应。
5. 缓存测试覆盖视频/采样身份变化、阈值仅重建、缺截图、损坏 JSON 与中断不标成功。
6. CLI/管线测试覆盖开关、显式阶段、无成片、依赖缺失、失败继续及独立命令退出码。
7. 报告测试覆盖转义；真实浏览器验证过滤、排序、全帧、分页及图片加载。
8. 真实模型用短成片验证四类输出、缓存复用与耗时；日常测试不下载模型。
9. 运行相关测试、完整默认 pytest 与受影响代码的 lint；全量测试不套 rtk。
10. 人工抽查命中与误报，不将单次试跑计数当作准确率或召回率。
