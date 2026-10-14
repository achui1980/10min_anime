# 内容检查实际验收（2026-10-14）

## 环境与素材

- macOS 26.7.1 arm64，Python 3.14.6，Homebrew FFmpeg 9.0.1。
- 模型：`deepghs/anime_dbrating` / `mobilenetv3_large_100_v0_ls0.2`，revision
  `7af21db648acdeb74f5c334abda9dd7403407b3c`，CPUExecutionProvider，4 线程。
- ONNX SHA-256：`c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362`。
- 实际项目成片：`/Users/portz/js/10min_anime/work/saijo/07_render/E02.mp4`，不是原片截取样片。
  全部输出及临时项目在 `feat/content-check` worktree 的 `work/content-check-validation/`。
  临时项目的最终视频为指向该成片的 symlink，原项目、配置和视频未修改。

## 运行与离线证据

```bash
uv run --extra content-check tenmin check-content \
  /Users/portz/js/10min_anime/work/saijo/07_render/E02.mp4 \
  --output work/content-check-validation/independent --offline
# 再运行一次，然后仅改阈值
uv run --extra content-check tenmin check-content \
  /Users/portz/js/10min_anime/work/saijo/07_render/E02.mp4 \
  --output work/content-check-validation/independent --offline --sensitive-threshold 0.6
uv run --extra content-check tenmin run saijo \
  --work-dir work/content-check-validation/projects --episode 2 --only check
```

scratch `validate.py` 用 Typer CliRunner 调用实际 CLI，两次独立输出目录推理与项目入口对比；
删除进程中的 API key，设置 `HF_HUB_OFFLINE=1`，对 socket connect 装失败哨兵。
以上检查在模型已缓存条件下全部通过，项目入口开关关闭仍执行，无需 API key。
这验证的是运行期离线；依赖安装/uv 解析并不在这个离线哨兵内。

| 检查 | 结果 |
|---|---|
| 首次/force 真推理 | 210 帧，24 候选；抽帧 2.484s，推理 2.667s，CLI 5.296s |
| 相同身份再运行 | 24 候选，0.042s，results.json 字节不变 |
| sensitive 0.5→0.6，adult 保持 0.5 | 18 候选，0.057s，原始分数/截图摘要和 mtime 不变 |
| 恢复默认阈值 | 24 候选，0.033s |
| 项目 only check | 210 帧，24 候选，4.992s；逐帧 time/image/label/scores 及候选范围完全一致 |

复跑和阈值重建阶段把 `extract_frames`、`score_frames` 替换成一旦调用就失败的哨兵，
因此“不推理”有调用证据，非仅根据耗时推测。

四类分数的本次最小/最大值（不是严重程度）：

| 类别 | 最小 | 最大 |
|---|---:|---:|
| general | 0.061910 | 0.713158 |
| sensitive | 0.108670 | 0.635078 |
| questionable | 0.056099 | 0.688526 |
| explicit | 0.060637 | 0.311532 |

## 真实媒体与浏览器

`tests/test_content_check_frames.py` 有 4 个常驻真实媒体回归：本地生成 320×240、10 fps、
3 秒 testsrc2；验证 SAR=2、90° 旋转、起始 PTS=5、音视频起点差。无模型/网络依赖，
缺 ffmpeg/ffprobe 则分别 skip。以 0.75 秒间隔抽样，实际播放时间为
`0, 0.8, 1.6, 2.4`；音视频偏移样片为 `0.2, 1.0, 1.8, 2.6`。
ffprobe 核对 JPEG 分别为 640×240、640×1706、640×240、640×480，SAR 均 1:1。
输入 ffprobe 同时确认真实旋转矩阵和 5 秒起点。FFmpeg 9 的 `rotate` metadata 未形成矩阵，
夹具改用输入 `-display_rotation 90`；没有修改生产抽帧或虚构序号时间。

Chrome DevTools 无法连接 localhost:9222，实际使用 Playwright 的
`chromium.launch(channel="chrome", headless=True)`（Chrome 155.0.8059.39）打开 file 报告。
默认 24 候选；两个阈值设为 1 后零候选；全帧模式从 60→120→210；时间排序首帧 00:00.00；
210 张图片均 `naturalWidth>0`；无 pageerror、无外部请求；交互前后 results.json SHA-256 相同。
`default-report.png` 留存实际页面，`browser-evidence.json` 留存断言结果。

## 人工观察与边界

人工查看最高排序候选 `186.186s`：男性赤膊上身、普通动漫人物表情，模型 top-1 为
questionable（0.688526），questionable+explicit 为 0.807294。
若目标是确认露骨色情，该画面是明显需要人工甄别的过宽候选，不能据分数确认。
普通帧 `0s` 是道路车辆画面，top-1 general（0.631490），未达默认候选阈值。
本次未建立人工完整标签集，未测准确率/召回率；泳装、裸露、色情的分类边界不能从一次试跑
推广。字幕、裁剪和动漫风格也可能影响图片评级。短于采样间隔的镜头可能漏掉，范围仅连接
采样点，不是精确片段起止。离线成功不等于模型判断正确。

## 回归、安装及证据位置

- 全量默认 pytest：2865 passed、35 skipped、1 个既有 google SDK deprecation warning。
- 受影响源代码/测试 Ruff clean；`uv lock --check` 和 `git diff --check` 通过。
- 非 editable wheel 构建并安装到独立基础环境；ONNX Runtime/NumPy/Pillow/HF hub 均未安装。
  三个 CLI help、默认关闭的 render→check fake 管线及 wheel 内 HTML 资源渲染通过，socket 哨兵通过。
  该环境相关 pytest 81 passed。
- 纯 Python JSON 转义断言已与可选 Node DOM 断言分开；逐集顺序断言只看该集 episode_start/end
  区间，不让前一集 render/check 掩盖后一集顺序问题。
- 默认 skip 的真实 API/ASR/OCR/render 测试不算本次验证通过；不覆盖未测试平台、极端 DAR、
  任意角旋转、损坏解码器、所有容器/编解码器或首次模型下载。
- 可复现脚本、JSON、截图、wheel、隔离环境与媒体测试输出在
  `work/content-check-validation/`；完整交付报告/Task8-only patch 在
  `.superpowers/sdd/2026-10-14-content-check/task-8-report.md` 和 `task-8-diff.patch`。
  这些 scratch 产物不随源码安装，运行证据路径是本机路径。
