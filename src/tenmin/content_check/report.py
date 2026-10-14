"""将检查结果渲染为离线 HTML；落盘由检查服务负责。"""

from importlib import resources

from tenmin.content_check.models import CheckManifest


def render_report(manifest: CheckManifest) -> str:
    template = resources.files("tenmin.content_check").joinpath("report.html").read_text(
        encoding="utf-8",
    )
    # HTML's script parser treats </script> as an end tag even inside JSON strings.
    data = manifest.model_dump_json().replace("<", "\\u003c")
    return template.replace("__DATA__", data)
