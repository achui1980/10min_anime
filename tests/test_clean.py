from tenmin.ingest.clean import apply_glossary, clean_text, strip_markup, to_simplified


def test_strip_markup_removes_ass_override_blocks():
    assert strip_markup(r"{\an8}{\fs20}你好") == "你好"


def test_strip_markup_removes_html_tags():
    assert strip_markup("<i>台词</i>") == "台词"
    assert strip_markup('<font color="#ffffff">台词</font>') == "台词"


def test_strip_markup_keeps_newlines_and_trims_each_line():
    assert strip_markup("  上句  \n  下句  ") == "上句\n下句"


def test_strip_markup_collapses_inner_spaces_but_keeps_cjk():
    assert strip_markup("你好\u3000世界") == "你好 世界"


def test_to_simplified():
    assert to_simplified("才女的侍從") == "才女的侍从"
    assert to_simplified("製作委員會") == "制作委员会"


def test_apply_glossary_runs_on_simplified_text():
    assert apply_glossary("伊月与大小姐", {"大小姐": "千金"}) == "伊月与千金"


def test_apply_glossary_longest_key_wins():
    glossary = {"天王寺": "TENNOJI", "天王寺璃子": "RIKO"}
    assert apply_glossary("天王寺璃子来了", glossary) == "RIKO来了"


def test_clean_text_pipeline_order():
    # 繁体 -> 简体 -> 术语表覆盖，术语表用简体 key 就能命中
    assert clean_text("才女的侍從", glossary={"侍从": "管家"}) == "才女的管家"


def test_clean_text_can_skip_conversion():
    assert clean_text("才女的侍從", convert=False) == "才女的侍從"


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""
