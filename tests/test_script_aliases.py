from tenmin.config import ProjectConfig
from tenmin.script.single import build_alias_block


def test_alias_block_lists_nicknames_with_real_name():
    assert build_alias_block({"猫猫": ["雀斑女", " 小猫 "]}) == "- 雀斑女、小猫 → 猫猫"


def test_alias_block_empty():
    assert build_alias_block({}) == "（无外号）"
    assert build_alias_block({"猫猫": [" "]}) == "（无外号）"


def test_project_config_accepts_aliases():
    cfg = ProjectConfig(show="s", slug="s", aliases={"猫猫": ["雀斑女"]})
    assert cfg.aliases == {"猫猫": ["雀斑女"]}
