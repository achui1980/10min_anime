"""从 animeko 公开 API 拉取番剧角色名，生成「日文名 → 简体中文名」人物名表。

只按条目 ID 拉取（不做搜索），结果缓存在 ``work/<slug>/meta/animeko-<id>.json``。
写回 project.yaml 的 ``glossary:`` 时只新增，不覆盖手写条目。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from ruamel.yaml.comments import CommentedMap

from tenmin import atomic

API_BASE = "https://api.animeko.org/v2"
USER_AGENT = "tenmin/glossary (https://github.com/)"
TIMEOUT_SECONDS = 30.0
# role 1 主角、2 配角；4/5 是店主、旁白、参与配音这类非角色条目。
CHARACTER_ROLES = frozenset({1, 2})


def cache_path(root: Path, subject: int) -> Path:
    return root / "meta" / f"animeko-{subject}.json"


def _get(client: httpx.Client, url: str) -> Any:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        raise ValueError(f"访问 animeko 失败：{url}：{exc or type(exc).__name__}") from exc
    if response.status_code != 200:
        raise ValueError(f"animeko 返回 HTTP {response.status_code}：{url}")
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(f"animeko 返回的不是 JSON：{url}") from exc


def fetch(subject: int, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """拉取条目与角色列表，返回 ``{"subject": ..., "characters": [...]}``。"""
    owned = client is None
    if client is None:
        client = httpx.Client(timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
    try:
        info = _get(client, f"{API_BASE}/subjects/{subject}")
        characters = _get(client, f"{API_BASE}/subjects/{subject}/characters")
    finally:
        if owned:
            client.close()
    if not isinstance(info, dict) or not isinstance(characters, list):
        raise ValueError(f"animeko 条目 {subject} 的数据格式认不出")
    return {"subject": info, "characters": characters}


def load(
    root: Path, subject: int, *, refresh: bool = False, client: httpx.Client | None = None
) -> dict[str, Any]:
    """优先读缓存；没有缓存或 ``refresh`` 时联网拉取并写缓存。"""
    path = cache_path(root, subject)
    if path.is_file() and not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("characters"), list):
            return data
    data = fetch(subject, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def build_names(characters: list[Any]) -> dict[str, str]:
    """只取主角/配角，跳过没有中文名或中文名与原名相同的条目。"""
    names: dict[str, str] = {}
    for item in characters:
        if not isinstance(item, dict) or item.get("role") not in CHARACTER_ROLES:
            continue
        character = item.get("character")
        if not isinstance(character, dict):
            continue
        name = character.get("name")
        name_cn = character.get("nameCn")
        if not isinstance(name, str) or not isinstance(name_cn, str):
            continue
        name, name_cn = name.strip(), name_cn.strip()
        if name and name_cn and name != name_cn and name not in names:
            names[name] = name_cn
    return names


@dataclass
class GlossaryPlan:
    added: dict[str, str] = field(default_factory=dict)
    same: dict[str, str] = field(default_factory=dict)
    # 日文名 → (现有中文名, animeko 中文名)
    conflicts: dict[str, tuple[str, str]] = field(default_factory=dict)


def plan(existing: dict[str, str], fetched: dict[str, str]) -> GlossaryPlan:
    result = GlossaryPlan()
    for name, name_cn in fetched.items():
        if name not in existing:
            result.added[name] = name_cn
        elif existing[name] == name_cn:
            result.same[name] = name_cn
        else:
            result.conflicts[name] = (existing[name], name_cn)
    return result


def write_glossary(yaml_path: Path, added: dict[str, str]) -> None:
    """把新增条目追加到 project.yaml 的 ``glossary:``，保留其余内容与注释。"""
    if not added:
        return
    from tenmin.pipeline import dump_round_trip, load_round_trip

    round_trip, data = load_round_trip(yaml_path)
    if data is None:
        data = CommentedMap()
    if not isinstance(data, CommentedMap):
        raise ValueError(f"{yaml_path} 顶层不是映射，无法写入 glossary")
    glossary = data.get("glossary")
    if glossary is None:
        glossary = CommentedMap()
        data["glossary"] = glossary
    if not isinstance(glossary, CommentedMap):
        raise ValueError(f"{yaml_path} 的 glossary 不是映射，无法写入")
    glossary.fa.set_block_style()
    for name, name_cn in added.items():
        if name not in glossary:
            glossary[name] = name_cn
    dump_round_trip(yaml_path, round_trip, data)
