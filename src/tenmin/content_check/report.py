"""将检查结果渲染为离线 HTML；落盘由检查服务负责。"""

from importlib import resources

from tenmin.content_check.models import CheckManifest


def render_report(manifest: CheckManifest, *, episode: int | None = None) -> str:
    """渲染报告页。

    `episode` 是这一集在项目里的集号，报告页用它给导出的遮罩标注文件命名、并在文件里
    记下集号（导入时会跟 `--episode` 对一遍，防止把 E02 的标注导进 E12）。独立命令
    `tenmin check-content` 不知道自己在检查第几集，这时传 None，导出时就不带这个字段，
    导入时由 CLI 补。
    """
    template = resources.files("tenmin.content_check").joinpath("report.html").read_text(
        encoding="utf-8",
    )
    # HTML's script parser treats </script> as an end tag even inside JSON strings.
    data = manifest.model_dump_json().replace("<", "\\u003c")
    html = template.replace("__DATA__", data)
    return html.replace("__EPISODE__", "null" if episode is None else str(episode))
