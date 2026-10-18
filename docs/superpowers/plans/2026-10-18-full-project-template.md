# 完整项目配置模板实现计划

**已批准设计：** init 展示全部 ProjectConfig 字段及嵌套默认值；episodes 保持空列表，注释展示全部 EpisodeConfig 属性。程序默认使用 minimax / MiniMax-M3、render.duck_db=-24.0。已有配置文件不改写，缺省字段遵循新默认值。

## 实施与验证

- [ ] 修改 tests/test_cli.py，要求生成 YAML 与完整 ProjectConfig.model_dump(mode="json") 相等，并验证三个新默认值。
- [ ] 运行 `uv run pytest tests/test_cli.py -q -k 'init or build_project_template'`，确认新要求失败。
- [ ] config.py 修改三个默认值；cli.py 删除模板字段白名单，直接完整 model_dump，更新模板头和单集注释。
- [ ] 更新旧默认值断言；专测 Gemini 的用例显式指定 provider="gemini"，保留 provider 行为验证。
- [ ] 更新 README.md / AGENTS.md，说明完整模板、缺省行为和新默认值。
- [ ] 运行配置、CLI、provider 相关测试和全量 `uv run pytest tests/ -q`，检查改动文件 Ruff 及 `git diff --check`。
