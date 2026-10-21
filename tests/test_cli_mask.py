import json

import yaml
from typer.testing import CliRunner

from tenmin.cli import app

runner = CliRunner()


def out(result) -> str:
    """click 8.2 起 result.output 只含 stdout，stderr 单独拿。这里合并，避免版本差异。"""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


def _bootstrap(work):
    root = work / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n\u4f60\u597d\n\n", encoding="utf-8"
    )
    (root / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "show": "\u624d\u5973\u7684\u4f8d\u4ece",
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


def _payload(**overrides):
    payload = {
        "schema_version": 1,
        "episode": 2,
        "sigma": 20.0,
        "ranges": [{"start": 156.0, "end": 162.5}],
    }
    payload.update(overrides)
    return payload


def _write_mask_file(work, payload, name="mask.json"):
    path = work / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _stored_mask(root):
    return root / "08_check" / "E02.mask.json"


def test_mask_status_without_annotation_points_at_the_report(tmp_path):
    root = _bootstrap(tmp_path)
    result = runner.invoke(app, ["mask", "saijo", "--episode", "2", "--work-dir", str(tmp_path)])
    assert result.exit_code == 0, out(result)
    text = out(result)
    assert "E02 \u6ca1\u6709\u906e\u7f69\u6807\u6ce8" in text
    assert "report.html" in text
    assert not _stored_mask(root).exists()


def test_mask_status_lists_every_range(tmp_path):
    root = _bootstrap(tmp_path)
    stored = _stored_mask(root)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text(
        json.dumps(
            _payload(ranges=[{"start": 156.0, "end": 162.5}, {"start": 300.0, "end": 301.0}]),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["mask", "saijo", "--episode", "2", "--work-dir", str(tmp_path)])
    assert result.exit_code == 0, out(result)
    text = out(result)
    assert "2 \u6bb5" in text
    assert "sigma=20" in text
    assert "00:02:36.000" in text
    assert "00:02:42.500" in text
    assert "00:05:00.000" in text
    assert str(stored) in text


def test_mask_import_stores_the_annotation_and_prints_the_render_command(tmp_path):
    root = _bootstrap(tmp_path)
    source = _write_mask_file(tmp_path, _payload())
    result = runner.invoke(
        app,
        [
            "mask",
            "saijo",
            "--episode",
            "2",
            "--work-dir",
            str(tmp_path),
            "--import",
            str(source),
        ],
    )
    assert result.exit_code == 0, out(result)
    text = out(result)
    assert "\u5df2\u5bfc\u5165 E02" in text
    assert "1 \u6bb5" in text
    assert "sigma=20" in text
    assert "tenmin run saijo --episode 2 --only render" in text
    stored = _stored_mask(root)
    assert stored.exists()
    saved = json.loads(stored.read_text(encoding="utf-8"))
    assert saved["episode"] == 2
    assert saved["sigma"] == 20.0
    assert saved["ranges"] == [{"start": 156.0, "end": 162.5}]


def test_mask_import_fills_in_the_episode_when_the_report_did_not_know_it(tmp_path):
    root = _bootstrap(tmp_path)
    payload = _payload()
    payload.pop("episode")
    source = _write_mask_file(tmp_path, payload)
    result = runner.invoke(
        app,
        [
            "mask",
            "saijo",
            "--episode",
            "2",
            "--work-dir",
            str(tmp_path),
            "--import",
            str(source),
        ],
    )
    assert result.exit_code == 0, out(result)
    saved = json.loads(_stored_mask(root).read_text(encoding="utf-8"))
    assert saved["episode"] == 2


def test_mask_import_rejects_an_annotation_for_another_episode(tmp_path):
    root = _bootstrap(tmp_path)
    source = _write_mask_file(tmp_path, _payload(episode=3))
    result = runner.invoke(
        app,
        [
            "mask",
            "saijo",
            "--episode",
            "2",
            "--work-dir",
            str(tmp_path),
            "--import",
            str(source),
        ],
    )
    assert result.exit_code == 1
    text = out(result)
    assert "\u4e0d\u80fd\u5bfc\u5165" in text
    assert not _stored_mask(root).exists()


def test_mask_import_reports_a_missing_file(tmp_path):
    _bootstrap(tmp_path)
    missing = tmp_path / "nope.json"
    result = runner.invoke(
        app,
        [
            "mask",
            "saijo",
            "--episode",
            "2",
            "--work-dir",
            str(tmp_path),
            "--import",
            str(missing),
        ],
    )
    assert result.exit_code == 1
    assert "\u627e\u4e0d\u5230\u8981\u5bfc\u5165\u7684\u6587\u4ef6" in out(result)


def test_mask_import_of_identical_content_keeps_the_file_untouched(tmp_path):
    root = _bootstrap(tmp_path)
    source = _write_mask_file(tmp_path, _payload())
    args = [
        "mask",
        "saijo",
        "--episode",
        "2",
        "--work-dir",
        str(tmp_path),
        "--import",
        str(source),
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, out(first)
    stored = _stored_mask(root)
    before = stored.stat().st_mtime_ns
    second = runner.invoke(app, args)
    assert second.exit_code == 0, out(second)
    assert "\u672a\u6539\u52a8" in out(second)
    assert stored.stat().st_mtime_ns == before


def test_mask_clear_removes_the_annotation(tmp_path):
    root = _bootstrap(tmp_path)
    source = _write_mask_file(tmp_path, _payload())
    imported = runner.invoke(
        app,
        [
            "mask",
            "saijo",
            "--episode",
            "2",
            "--work-dir",
            str(tmp_path),
            "--import",
            str(source),
        ],
    )
    assert imported.exit_code == 0, out(imported)
    assert _stored_mask(root).exists()
    result = runner.invoke(
        app,
        ["mask", "saijo", "--episode", "2", "--work-dir", str(tmp_path), "--clear"],
    )
    assert result.exit_code == 0, out(result)
    assert "\u5df2\u5220\u9664" in out(result)
    assert not _stored_mask(root).exists()


def test_mask_clear_without_annotation_says_nothing_to_do(tmp_path):
    root = _bootstrap(tmp_path)
    result = runner.invoke(
        app,
        ["mask", "saijo", "--episode", "2", "--work-dir", str(tmp_path), "--clear"],
    )
    assert result.exit_code == 0, out(result)
    assert "\u65e0\u9700\u5220\u9664" in out(result)
    assert not _stored_mask(root).exists()


def test_mask_rejects_an_episode_that_is_not_registered(tmp_path):
    _bootstrap(tmp_path)
    result = runner.invoke(app, ["mask", "saijo", "--episode", "7", "--work-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert out(result).strip() != ""
