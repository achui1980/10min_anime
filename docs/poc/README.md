# 动漫分级 POC（可丢弃实验）

问题：现成的动漫图片分级模型，能否筛出番剧里的非裸露福利／性暗示画面？

使用 DeepGHS `anime_dbrating` 的 ONNX 权重，本地 CPU 推理；视频、截图不会上传。
首次运行需要联网下载 Python 依赖与模型。模型版本固定为
`7af21db648acdeb74f5c334abda9dd7403407b3c`。

## 运行

在仓库根目录执行（需要 uv、ffmpeg）：

```bash
uv run docs/poc/anime_rating_poc.py work/saijo/video/E02.mp4 \
  --model mobile --interval 1 --output work/nsfw-poc-saijo-e02
open work/nsfw-poc-saijo-e02/report.html
```

`--output` 必须是尚不存在的目录，每次实验用不同目录。可用 `--start 300 --duration 120`
只检查指定两分钟。`--interval` 是采样间隔秒数，越小越慢、越不容易漏掉短镜头。

可选模型：`swin`（默认，权重约 418MB）、`caformer`（约 150MB）、`mobile`（约 17MB）。
建议先用已实测的 `mobile` 验证流程；本次环境中 `swin` 权重下载超时，未完成试跑。
下载后加 `--offline` 只读模型缓存；完全离线重跑可使用
`uv run --offline docs/poc/anime_rating_poc.py ... --offline`（需先缓存该脚本环境与模型）。

## 看报告

- `report.html`：本地打开，显示时间点、截图与四级分数；可调阈值、查看全部采样帧。
- `results.json`：所有采样帧的分数、真实采样 PTS、权重摘要、推理耗时。
- `frames/`：采样截图；报告依赖此目录，移动报告时一起移动。
- 候选规则：`sensitive >= 0.5` 或 `questionable + explicit >= 0.5`，仅是可调的实验起点。
- 相邻候选采样点合并显示，其范围不是精确的视频片段边界。

预处理遵循官方 imgutils：RGB、双线性缩放到模型输入尺寸、CHW、归一化到 [-1, 1]。
为缩小报告体积，模型输入来自宽 640 的 JPEG 截图，可能不同于直接输入原图的结果。

## 验证有效性

不要把运行成功当作检测有效。先人工选出一些目标镜头与普通镜头，查看全部采样帧，
对照命中与漏报，记录误报和漏报；阈值应在不同片段上验证。

评级来自动漫插画数据，不等于“色情”；`sensitive` 也可能只是普通性感穿着。
模型不读取台词、不理解连续动作；未采样到的画面无法检测。
本实验不自动删片、打码，也没有经过番剧数据集准确率验证。

## 本机实测（2026-10-14）

素材：`work/saijo/video/E02.mp4`，约 23 分 40 秒；模型 `mobile`，CPUExecutionProvider，4 线程。
耗时仅包含抽帧与逐帧推理，不包含首次依赖／模型下载及模型加载。

| 采样间隔 | 帧数 | 抽帧 | 推理 | 默认阈值候选 |
| --- | ---: | ---: | ---: | ---: |
| 5 秒 | 284 | 10.8 秒 | 2.2 秒 | 26 |
| 1 秒 | 1419 | 11.5 秒 | 10.8 秒 | 127 |

已生成报告：
- `work/nsfw-poc-saijo-e02-mobile/report.html`（5 秒）
- `work/nsfw-poc-saijo-e02-mobile-1s/report.html`（1 秒，建议先看这个）

人工抽查高分截图：能标出泳装、泳池福利镜头；也会把泳池中男性的普通脸部特写
打成高分候选。1 秒采样中 19:33.19 的泳装腰臀特写被评级为 `explicit`，
但截图没有可见的私密部位裸露。这证明评级不能直接用作“是否裸露”的判据。
尚未逐帧标注真值，不能据此给出准确率或召回率。

本地 Chrome 无头验证：截图可加载、阈值过滤、全帧查看、排序与分页正常，未发现 JS 错误。
结论：值得作为人工审核的候选生成器继续评估，不足以自动判定或删除性暗示片段。

来源：
- https://huggingface.co/deepghs/anime_dbrating
- https://github.com/deepghs/imgutils/blob/main/imgutils/generic/classify.py
