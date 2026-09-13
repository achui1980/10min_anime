import pytest
import yaml
from typer.testing import CliRunner

from tenmin.cli import app
from tenmin.render.ffmpeg import FFmpegError

runner = CliRunner()


def out(result) -> str:
    """click 8.2 起 result.output 只含 stdout，stderr 单独拿。这里合并，避免版本差异。"""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


def test_init_creates_project_yaml(work):
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code == 0, out(result)
    path = work / "saijo" / "project.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["slug"] == "saijo"
    assert data["mode"] == "single_episode"
    assert data["target_seconds"] == 240


def test_init_creates_srt_dir(work):
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert (work / "saijo" / "srt").is_dir()


def test_init_refuses_to_clobber(work):
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "已存在" in out(result)


def test_run_missing_project_fails(work):
    result = runner.invoke(app, ["run", "nope", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "project.yaml" in out(result)


def test_run_unknown_stage_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "nope"]
    )
    assert result.exit_code != 0


def test_run_unknown_from_stage_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--from", "nope"]
    )
    assert result.exit_code != 0
    assert "未知阶段" in out(result)


def test_run_only_accepts_repeated_flags(work, golden_srt_path):
    """--only 原先只收一个阶段（CLI 传 str），而 run_pipeline 的 only 本来就吃序列。"""
    root = _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app,
        [
            "run", "saijo", "--work-dir", str(work),
            "--only", "ingest", "--only", "signals",
        ],
    )
    assert result.exit_code == 0, out(result)
    assert (root / "01_dialogue" / "E02.dialogue.json").exists()
    assert (root / "02_signals" / "E02.signals.json").exists()
    assert not (root / "03_script").exists()


def test_run_only_accepts_comma_separated_stages(work, golden_srt_path):
    root = _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest,signals"]
    )
    assert result.exit_code == 0, out(result)
    assert (root / "01_dialogue" / "E02.dialogue.json").exists()
    assert (root / "02_signals" / "E02.signals.json").exists()


def test_run_only_rejects_unknown_stage_among_valid_ones(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest,nope"]
    )
    assert result.exit_code != 0
    assert "nope" in out(result)


def test_run_only_passes_every_stage_to_the_pipeline(tmp_path, monkeypatch):
    """CLI 校验用的阶段列表与真正传给 run_pipeline 的 only 必须是同一份。"""
    _minimal_project(tmp_path)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    result = runner.invoke(
        app,
        ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen,timeline"],
    )
    assert result.exit_code == 0, out(result)
    assert captured["only"] == ["docgen", "timeline"]


def test_run_without_only_passes_none(tmp_path, monkeypatch):
    _minimal_project(tmp_path)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    monkeypatch.setattr("tenmin.cli.build_tts_engine", lambda cfg: object())
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())
    result = runner.invoke(app, ["run", "akujo2", "--work-dir", str(tmp_path)])
    assert result.exit_code == 0, out(result)
    assert captured["only"] is None


def _bootstrap(work, golden_srt_path):
    root = work / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    (root / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "show": "才女的侍从",
                "slug": "saijo",
                "mode": "single_episode",
                "target_seconds": 240,
                "episodes": [{"number": 2, "srt": "srt/E02.srt"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root


def test_run_only_ingest_needs_no_api_key(work, golden_srt_path, monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    root = _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"]
    )
    assert result.exit_code == 0, out(result)
    assert (root / "01_dialogue" / "E02.dialogue.json").exists()


def test_run_script_stage_without_api_key_fails(work, golden_srt_path, monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(app, ["run", "saijo", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "TENMIN_GEMINI_API_KEY" in out(result)


def test_inspect_prints_summary(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "signals"])
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2"]
    )
    assert result.exit_code == 0, out(result)
    text = out(result)
    assert "对白轨" in text
    assert "无字幕间隙" in text
    assert "153.486" in text


def test_inspect_shows_suspect_lines(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "signals"])
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2", "--suspect"]
    )
    assert result.exit_code == 0, out(result)
    assert "80-08" in out(result)


def test_inspect_without_ingest_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2"]
    )
    assert result.exit_code != 0


def _minimal_project(tmp_path):
    root = tmp_path / "akujo2"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 我是不才恶女\n"
        "slug: akujo2\n"
        "episodes:\n"
        "- number: 2\n"
        "  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    return root


def test_init_template_carries_render_defaults(tmp_path):
    result = runner.invoke(app, ["init", "akujo2", "--work-dir", str(tmp_path)])

    assert result.exit_code == 0
    data = yaml.safe_load(
        (tmp_path / "akujo2" / "project.yaml").read_text(encoding="utf-8")
    )
    assert data["render"]["voice"] == "zh-CN-YunxiNeural"
    assert data["render"]["rate"] == "+0%"
    assert data["render"]["video_encoder"] == "libx264"
    assert data["render"]["duck_db"] == -12.0
    assert data["render"]["font_size"] == 52
    assert data["render"]["fade_out_seconds"] == 1.5
    assert data["render"]["outro_card_seconds"] == 3.0
    assert data["render"]["outro_message"] == "解说结束，谢谢观看"


def test_init_does_not_create_a_video_dir(work):
    """源片不会被拷进 work/，register_episode 只记它的绝对路径。

    一个空的 work/<slug>/video/ 夹在 01_dialogue/…07_render/ 中间，读起来就是
    「源片放这里」，而 project.yaml 里明明指向别的盘，纯属自相矛盾的误导。
    想用相对路径手动放片的人自己 mkdir 就行，没有任何代码依赖这个目录预先存在。
    """
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code == 0, out(result)
    assert (work / "saijo" / "srt").is_dir()
    assert not (work / "saijo" / "video").exists()


def test_init_tells_user_how_to_register_an_episode(work):
    """原文案是「把字幕放进 …/srt、源视频放进 …/video」，而源片压根不进 work/。"""
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    text = out(result)
    assert "tenmin run saijo --episode" in text
    assert "--srt" in text and "--video" in text
    assert "源视频放进" not in text


def test_init_template_follows_the_model_defaults(work, monkeypatch):
    """模板不许跟 config.py 平行维护：改模型的默认值，init 生成的 yaml 必须跟着变。

    原实现是手抄一份字面量的 PROJECT_TEMPLATE dict，改 RenderConfig 的默认值时
    这份拷贝不会动，`tenmin init` 生成的 project.yaml 就静默落后于代码。
    """
    from pydantic import Field

    from tenmin.config import LLMConfig, ProjectConfig, RenderConfig

    class ShiftedRender(RenderConfig):
        font_size: int = 77
        outro_message: str = "改过的片尾语"

    class ShiftedLLM(LLMConfig):
        model: str = "改过的模型名"

    class ShiftedProject(ProjectConfig):
        llm: ShiftedLLM = Field(default_factory=ShiftedLLM)
        render: ShiftedRender = Field(default_factory=ShiftedRender)

    monkeypatch.setattr("tenmin.cli.ProjectConfig", ShiftedProject)
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code == 0, out(result)

    data = yaml.safe_load((work / "saijo" / "project.yaml").read_text(encoding="utf-8"))
    assert data["render"]["font_size"] == 77
    assert data["render"]["outro_message"] == "改过的片尾语"
    assert data["llm"]["model"] == "改过的模型名"


def test_init_output_round_trips_through_load_project(work):
    """init 生成的 yaml 必须能被 load_project 原样读回来，且逐字段等于模型默认值。"""
    from tenmin.config import ProjectConfig, load_project

    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code == 0, out(result)

    cfg = load_project(work / "saijo" / "project.yaml")
    expected = ProjectConfig(show="saijo", slug="saijo")
    assert cfg.model_dump() == expected.model_dump()


def test_init_output_has_no_keys_the_models_do_not_know(work):
    """独立于 pydantic 的 extra 设置，直接查生成的 yaml 里有没有模型不认识的键。

    config.py 的模型目前是 pydantic 默认的 extra="ignore"，写错的键会被静默吞掉，
    所以「能 load 回来」并不等于「每个键都真的生效」。这条自己查。
    """
    from tenmin.config import LLMConfig, LocaleConfig, ProjectConfig, RenderConfig

    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    data = yaml.safe_load((work / "saijo" / "project.yaml").read_text(encoding="utf-8"))

    assert set(data) <= set(ProjectConfig.model_fields)
    assert set(data["llm"]) <= set(LLMConfig.model_fields)
    assert set(data["render"]) <= set(RenderConfig.model_fields)
    assert set(data["locale"]) <= set(LocaleConfig.model_fields)


def test_init_leaves_episodes_empty(work):
    """模板不许预置示例 episode。

    register_episode 是「读回 cfg.episodes 再整份写回」的，示例条目会被当成一集真的
    番留在 yaml 里：`tenmin run <slug> --episode 1 --srt X --video Y` 之后 episodes
    变成 [示例 E02, 真的 E01]，接着批处理模式（不带 --episode）就会去跑那个指向
    srt/E02.srt 的幽灵条目并崩掉。
    """
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    data = yaml.safe_load((work / "saijo" / "project.yaml").read_text(encoding="utf-8"))
    assert data["episodes"] == []


def test_init_yaml_explains_how_to_register_episodes(work):
    """episodes 是空的，那「怎么加一集」必须在文件里说清楚。"""
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    text = (work / "saijo" / "project.yaml").read_text(encoding="utf-8")
    assert "tenmin run saijo --episode" in text
    assert "--srt" in text and "--video" in text


def test_build_project_template_returns_a_fresh_object_each_call():
    """原实现是 dict(PROJECT_TEMPLATE) 浅拷贝：嵌套的 episodes/render/llm 与模块级
    常量共享同一个对象，任何对嵌套字段的改写都会污染全局。"""
    from tenmin.cli import build_project_template
    from tenmin.config import RenderConfig

    first = build_project_template("a")
    second = build_project_template("b")

    first["episodes"].append({"number": 99})
    first["render"]["font_size"] = 999
    first["llm"]["model"] = "污染"

    assert second["episodes"] == []
    assert second["render"]["font_size"] == RenderConfig().font_size
    assert second["llm"]["model"] != "污染"

    third = build_project_template("c")
    assert third["episodes"] == []
    assert third["render"]["font_size"] == RenderConfig().font_size


def test_build_project_template_values_all_come_from_the_models():
    """逐字段比对：模板里出现的每个值都必须等于对应模型的默认值。

    这条挡的是「后来人图省事又塞回一个字面量」。
    """
    from tenmin.cli import build_project_template
    from tenmin.config import LLMConfig, LocaleConfig, RenderConfig

    template = build_project_template("saijo")
    for section, model in (
        ("llm", LLMConfig()),
        ("render", RenderConfig()),
        ("locale", LocaleConfig()),
    ):
        for key, value in template[section].items():
            assert value == getattr(model, key), f"{section}.{key}"


def test_run_passes_tts_engine_when_voice_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)
    sentinel = object()
    monkeypatch.setattr("tenmin.cli.build_tts_engine", lambda cfg: sentinel)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "voice"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is sentinel


def test_run_skips_tts_engine_when_voice_not_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    def boom(cfg):
        raise AssertionError("只跑 docgen 不该造 TTS engine")

    monkeypatch.setattr("tenmin.cli.build_tts_engine", boom)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is None


def test_run_reports_ffmpeg_error(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise FFmpegError("你的 ffmpeg 没编 libass")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 1
    assert "没编 libass" in result.output


def _graceful(result) -> bool:
    """CliRunner 对「typer.Exit(1)」和「异常逃到顶层」都给 exit_code == 1，
    只能靠 result.exception 区分：优雅退出是 SystemExit，裸 traceback 是原异常本身。"""
    return result.exception is None or isinstance(result.exception, SystemExit)


def test_run_reports_script_validation_error(tmp_path, monkeypatch):
    """ScriptValidationError 是 RuntimeError 子类，不在原捕获列表里，
    于是 LLM 出的剧本过不了校验时用户看到一整页 traceback。"""
    from tenmin.script.validate import ScriptValidationError

    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise ScriptValidationError("剧本只有 2 个 beat，少于 3 个")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "少于 3 个" in out(result)


def test_run_reports_http_status_error(tmp_path, monkeypatch):
    """provider 的 response.raise_for_status() 抛的 httpx.HTTPStatusError（429/5xx）
    原先直接冒到顶层。"""
    import httpx

    _minimal_project(tmp_path)
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(429, request=request, text="rate limited")

    async def boom(cfg, provider, **kwargs):
        raise httpx.HTTPStatusError(
            "Client error '429 Too Many Requests'", request=request, response=response
        )

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "429" in out(result)


def test_run_reports_http_transport_error(tmp_path, monkeypatch):
    """连不上/读超时走的是 httpx.TransportError，跟 HTTPStatusError 一样该被兜住。"""
    import httpx

    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "Connection refused" in out(result)


def test_run_closes_the_provider_connection_pool(tmp_path, monkeypatch):
    """provider 现在持有一个 httpx.AsyncClient，跑完必须有人收尾。"""
    _minimal_project(tmp_path)
    closed: list[bool] = []

    class FakeProviderWithPool:
        async def aclose(self):
            closed.append(True)

    async def ok(cfg, provider, **kwargs):
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", ok)
    monkeypatch.setattr(
        "tenmin.cli.build_provider", lambda llm, settings: FakeProviderWithPool()
    )

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 0, out(result)
    assert closed == [True]


def test_run_closes_the_provider_even_when_the_pipeline_blows_up(tmp_path, monkeypatch):
    _minimal_project(tmp_path)
    closed: list[bool] = []

    class FakeProviderWithPool:
        async def aclose(self):
            closed.append(True)

    async def boom(cfg, provider, **kwargs):
        raise FileNotFoundError("缺少对白轨产物")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr(
        "tenmin.cli.build_provider", lambda llm, settings: FakeProviderWithPool()
    )

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert closed == [True]


def test_run_reports_llm_error(tmp_path, monkeypatch):
    """LLMError 是 RuntimeError 子类（不在 ValueError 那条网里），provider 抛的
    「HTTP 4xx 带响应体」「业务错误码」「连续 N 次不合 schema」全走它。"""
    from tenmin.script.llm import LLMError

    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise LLMError("LLM 接口返回 HTTP 401：响应体：invalid api key")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "invalid api key" in out(result)


def test_run_reports_error_type_when_message_is_empty(tmp_path, monkeypatch):
    """httpx 的传输类异常经常 str() 为空，光 secho(str(error)) 会印一行空红字。"""
    import httpx

    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise httpx.ReadTimeout("")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "script"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "ReadTimeout" in out(result)


def test_run_reports_pydantic_validation_error(tmp_path, monkeypatch):
    """pydantic 的 ValidationError 是 ValueError 子类，已被现有捕获列表覆盖——
    这条只是把这个「已经没事」的事实钉住，免得后来人把 ValueError 换成更窄的类型。"""
    from pydantic import ValidationError

    from tenmin.models import Beat

    _minimal_project(tmp_path)
    try:
        Beat.model_validate({})
    except ValidationError as exc:
        captured = exc

    assert isinstance(captured, ValueError)

    async def boom(cfg, provider, **kwargs):
        raise captured

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"]
    )

    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)


def test_run_prints_mp4_path(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    mp4 = root / "07_render" / "E02.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"")

    async def fake_pipeline(cfg, provider, **kwargs):
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 0
    assert "成品视频" in result.output
    assert str(mp4) in result.output


def test_run_srt_without_video_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--episode",
            "1",
            "--srt",
            str(golden_srt_path),
        ],
    )
    assert result.exit_code != 0
    assert "--srt" in out(result) and "--video" in out(result)


def test_run_srt_video_without_episode_fails(work, golden_srt_path, tmp_path):
    _bootstrap(work, golden_srt_path)
    fake_video = tmp_path / "E01.mp4"
    fake_video.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--srt",
            str(golden_srt_path),
            "--video",
            str(fake_video),
        ],
    )
    assert result.exit_code != 0
    assert "--episode" in out(result)


def test_run_episode_not_registered_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app,
        ["run", "saijo", "--work-dir", str(work), "--episode", "99"],
    )
    assert result.exit_code != 0
    assert "没有注册" in out(result)


def test_run_registers_new_episode_and_updates_yaml(work, golden_srt_path, tmp_path):
    _bootstrap(work, golden_srt_path)
    fake_video = tmp_path / "E01_source.mp4"
    fake_video.write_bytes(b"fake video bytes")

    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--episode",
            "1",
            "--srt",
            str(golden_srt_path),
            "--video",
            str(fake_video),
            "--only",
            "ingest",
        ],
    )

    assert result.exit_code == 0, out(result)
    assert (work / "saijo" / "srt" / "E01.srt").exists()
    # 源片不再被整份拷进项目目录，project.yaml 直接记它的绝对路径。
    assert not (work / "saijo" / "video" / "E01.mp4").exists()

    reloaded = yaml.safe_load((work / "saijo" / "project.yaml").read_text(encoding="utf-8"))
    entry = next(e for e in reloaded["episodes"] if e["number"] == 1)
    assert entry["video"] == str(fake_video.resolve())


def test_run_passes_progress_reporter(tmp_path, monkeypatch):
    from tenmin.progress import ProgressReporter

    _minimal_project(tmp_path)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"]
    )
    assert result.exit_code == 0, out(result)
    assert isinstance(captured["reporter"], ProgressReporter)
