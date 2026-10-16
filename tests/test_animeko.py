import json

import httpx
import pytest
from typer.testing import CliRunner

from tenmin import animeko
from tenmin.cli import app

SUBJECT = {"id": 568244, "name": "薬屋のひとりごと 第3期", "nameCn": "药屋少女的呢喃 第三季"}
CHARACTERS = [
    {"role": 1, "character": {"id": 1, "name": "猫猫", "nameCn": "猫猫"}},
    {"role": 1, "character": {"id": 2, "name": "壬氏", "nameCn": "壬氏改"}},
    {"role": 2, "character": {"id": 3, "name": "高順", "nameCn": "高顺"}},
    {"role": 2, "character": {"id": 4, "name": "趙迂", "nameCn": "赵迂"}},
    {"role": 2, "character": {"id": 5, "name": "二番", "nameCn": ""}},
    {"role": 4, "character": {"id": 6, "name": "店主", "nameCn": "店长"}},
    {"role": 5, "character": {"id": 7, "name": "ナレーション", "nameCn": "旁白"}},
]


def _transport(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/characters"):
            return httpx.Response(200, json=CHARACTERS)
        return httpx.Response(200, json=SUBJECT)

    return httpx.MockTransport(handler)


def test_build_names_keeps_only_characters_with_distinct_chinese():
    assert animeko.build_names(CHARACTERS) == {"壬氏": "壬氏改", "高順": "高顺", "趙迂": "赵迂"}


def test_plan_never_overwrites_existing():
    existing = {"高順": "高顺", "壬氏": "壬氏"}
    result = animeko.plan(existing, {"高順": "高顺", "壬氏": "壬氏改", "趙迂": "赵迂"})
    assert result.added == {"趙迂": "赵迂"}
    assert result.same == {"高順": "高顺"}
    assert result.conflicts == {"壬氏": ("壬氏", "壬氏改")}


def test_load_caches_and_refresh(tmp_path):
    calls = []
    client = httpx.Client(transport=_transport(calls))
    data = animeko.load(tmp_path, 568244, client=client)
    assert data["subject"]["id"] == 568244
    assert len(calls) == 2
    assert animeko.cache_path(tmp_path, 568244).is_file()
    animeko.load(tmp_path, 568244, client=client)
    assert len(calls) == 2
    animeko.load(tmp_path, 568244, refresh=True, client=client)
    assert len(calls) == 4


def test_http_error_is_chinese_value_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(ValueError, match="HTTP 404"):
        animeko.fetch(1, client=client)


def test_network_error_is_chinese_value_error():
    def boom(request):
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    with pytest.raises(ValueError, match="访问 animeko 失败"):
        animeko.fetch(1, client=client)


def test_write_glossary_preserves_comments_and_existing(tmp_path):
    project = tmp_path / "project.yaml"
    project.write_text(
        "# 我的番\ntitle: 测试  # 标题\nglossary:\n  高順: 高顺\nepisodes: []\n",
        encoding="utf-8",
    )
    animeko.write_glossary(project, {"趙迂": "赵迂"})
    text = project.read_text(encoding="utf-8")
    assert "# 我的番" in text and "# 标题" in text
    assert "高順: 高顺" in text and "趙迂: 赵迂" in text


def test_write_glossary_creates_section_from_flow_empty(tmp_path):
    project = tmp_path / "project.yaml"
    project.write_text("title: 测试\nglossary: {}\n", encoding="utf-8")
    animeko.write_glossary(project, {"趙迂": "赵迂"})
    assert "glossary:\n  趙迂: 赵迂" in project.read_text(encoding="utf-8")


def _project(tmp_path, glossary="{}"):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "project.yaml").write_text(
        f"show: 测试\nslug: demo\nglossary: {glossary}\nepisodes: []\n", encoding="utf-8"
    )
    cache = animeko.cache_path(root, 568244)
    cache.parent.mkdir()
    cache.write_text(json.dumps({"subject": SUBJECT, "characters": CHARACTERS}), encoding="utf-8")
    return root


def test_cli_dry_run_does_not_write(tmp_path):
    root = _project(tmp_path)
    before = (root / "project.yaml").read_text(encoding="utf-8")
    result = CliRunner().invoke(
        app, ["glossary", "demo", "--subject", "568244", "--work-dir", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "新增  趙迂 → 赵迂" in result.output
    assert (root / "project.yaml").read_text(encoding="utf-8") == before


def test_cli_yes_writes_and_keeps_conflicts(tmp_path):
    root = _project(tmp_path, glossary="{壬氏: 壬氏}")
    result = CliRunner().invoke(
        app, ["glossary", "demo", "--subject", "568244", "--work-dir", str(tmp_path), "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "保留  壬氏 → 壬氏" in result.output
    from tenmin.config import load_project

    assert load_project(root / "project.yaml").glossary == {
        "壬氏": "壬氏", "高順": "高顺", "趙迂": "赵迂"
    }


def test_cli_non_tty_without_yes_fails(tmp_path, monkeypatch):
    _project(tmp_path)
    monkeypatch.setattr("tenmin.cli._stdin_is_interactive", lambda: False)
    result = CliRunner().invoke(
        app, ["glossary", "demo", "--subject", "568244", "--work-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
