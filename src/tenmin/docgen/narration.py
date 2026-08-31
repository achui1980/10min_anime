"""合并配音纯文本。直接喂 TTS，所以不带任何标记符号。"""

from __future__ import annotations

from tenmin.models import Script


def render_narration(script: Script) -> str:
    chunks = [beat.narration.strip() for beat in script.beats if beat.narration.strip()]
    if not chunks:
        return ""
    return "\n\n".join(chunks) + "\n"
