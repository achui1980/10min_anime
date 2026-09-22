"""累积术语表。纯文件读写与字典合并，没有外部依赖。"""

import os

from tenmin.translate import glossary as g


def test_loading_a_missing_file_gives_an_empty_table(tmp_path):
    """第一集跑之前这个文件不存在，那不是错误。"""
    assert g.load_glossary(tmp_path / "nope.json") == {}


def test_loading_a_directory_gives_an_empty_table(tmp_path):
    """路径存在但不是文件（手动建了个同名目录）走的是同一条「没有可用累积表」分支。"""
    path = tmp_path / "glossary.json"
    path.mkdir()
    assert g.load_glossary(path) == {}


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "zh" / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"})
    assert g.load_glossary(path) == {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}


def test_save_creates_the_parent_directory(tmp_path):
    """`zh/` 在第一次存盘时还不存在（没有别的阶段会去建它）。"""
    path = tmp_path / "zh" / "glossary.json"
    g.save_glossary(path, {"イ": "伊"})
    assert path.is_file()


def test_save_leaves_no_part_file_behind(tmp_path):
    """守住「产物目录里不许留临时文件」：save 之后那个目录里只有正式产物一个文件。

    刻意**不要**把这条读成「守住了原子写」。实测证伪过：把 `save_glossary` 里的
    `atomic.write_text` 换成裸 `path.write_text`（完全绕过原子写），本文件照样 26 passed
    全绿 —— 普通 `write_text` 同样不留 `.part`，所以这条断言在原理上钉不住原子性。原子写
    那条不变量由 `test_source_hygiene.py` 的 `test_artifact_writes_go_through_atomic`
    守着（那次变异实验里只有它变红）。

    两条各管一半、都不该删：那条审计是**语法**的，能抓到「调用点绕过了 atomic」这种这里
    测不出来的形态，但它看不出 `atomic` 自己还好不好使；而这条是运行时的，`atomic_path`
    哪天不 replace 了、或者 save 开始在旁边留个备份文件，只有它会红。
    """
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"イ": "伊"})
    assert [p.name for p in tmp_path.iterdir()] == ["glossary.json"]


def test_saved_file_is_human_readable_utf8(tmp_path):
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚"})
    body = path.read_text(encoding="utf-8")
    assert "莉迪亚" in body
    assert "\\u" not in body


def test_saved_keys_are_sorted(tmp_path):
    """稳定顺序让这个文件的 diff 可读 —— 它是人会去手动纠错的文件。"""
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"ロ": "罗", "イ": "伊", "ハ": "哈"})
    body = path.read_text(encoding="utf-8")
    assert body.index('"イ"') < body.index('"ハ"') < body.index('"ロ"')


def test_saved_file_ends_with_a_newline(tmp_path):
    """人会用编辑器改它，缺行尾换行会让 diff 多出一条 `\\ No newline` 噪声。"""
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"イ": "伊"})
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_loading_a_corrupt_file_gives_an_empty_table(tmp_path):
    """这个文件坏了不该让整条管线停 —— 最坏情况是这一集的译名不跟前几集对齐。"""
    path = tmp_path / "glossary.json"
    path.write_text("{ 这不是 json", encoding="utf-8")
    assert g.load_glossary(path) == {}


def test_loading_a_file_that_is_not_utf8_gives_an_empty_table(tmp_path):
    """被 cp932 编辑器存回去的表。UnicodeDecodeError 不是 JSONDecodeError 的亲戚，
    单捕后者会让这条路径把异常漏出去、停掉整条管线。"""
    path = tmp_path / "glossary.json"
    path.write_bytes(b'{"\x83\x8a\x83f\x83B\x83A": "x"}')
    assert g.load_glossary(path) == {}


def test_loading_a_json_list_gives_an_empty_table(tmp_path):
    """合法 JSON 但不是对象。少了这条判断会在 `.items()` 上炸 AttributeError。"""
    path = tmp_path / "glossary.json"
    path.write_text('["リディア"]', encoding="utf-8")
    assert g.load_glossary(path) == {}


def test_loading_drops_entries_whose_translation_is_not_a_string(tmp_path):
    """手动改坏一条不该污染另外几条，也不该把 `null` 变成译名 "None" 喂进 prompt。"""
    path = tmp_path / "glossary.json"
    path.write_text('{"リディア": "莉迪亚", "ルーファス": null, "ハ": 3}', encoding="utf-8")
    assert g.load_glossary(path) == {"リディア": "莉迪亚"}


def test_loading_drops_blank_entries(tmp_path):
    """留了个空壳条目（人手删了译名但没删键）等于没这条。"""
    path = tmp_path / "glossary.json"
    path.write_text('{"リディア": "  ", "  ": "伊"}', encoding="utf-8")
    assert g.load_glossary(path) == {}


def test_merge_keeps_the_accumulated_choice_for_known_terms():
    """已经定下的译名不许被后面某一集改掉 —— 那正是累积要防的事。"""
    merged = g.merge_glossary({"リディア": "莉迪亚"}, {"リディア": "莉蒂亚"})
    assert merged == {"リディア": "莉迪亚"}


def test_merge_adds_new_terms():
    merged = g.merge_glossary({"リディア": "莉迪亚"}, {"ルーファス": "鲁弗斯"})
    assert merged == {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}


def test_merge_ignores_blank_entries():
    merged = g.merge_glossary({}, {"リディア": "", "": "空键", "  ": "  "})
    assert merged == {}


def test_merge_ignores_a_blank_entry_on_the_accumulated_side():
    """两边都要洗，否则一条坏的累积条目会永远占着那个键、把好的新译名挡在外面。"""
    merged = g.merge_glossary({"リディア": "  "}, {"リディア": "莉迪亚"})
    assert merged == {"リディア": "莉迪亚"}


def test_merge_strips_whitespace():
    merged = g.merge_glossary({}, {" リディア ": " 莉迪亚 "})
    assert merged == {"リディア": "莉迪亚"}


def test_merge_does_not_mutate_its_inputs():
    accumulated = {"リディア": "莉迪亚"}
    fresh = {"ルーファス": "鲁弗斯"}
    g.merge_glossary(accumulated, fresh)
    assert accumulated == {"リディア": "莉迪亚"}
    assert fresh == {"ルーファス": "鲁弗斯"}


def test_manual_entries_override_accumulated_ones():
    """手写表是纠错入口：机器译错了人要能盖掉它。"""
    effective = g.effective_glossary({"リディア": "莉蒂亚"}, {"リディア": "莉迪亚"})
    assert effective["リディア"] == "莉迪亚"


def test_effective_glossary_unions_both_sides():
    effective = g.effective_glossary({"イ": "伊"}, {"ロ": "罗"})
    assert effective == {"イ": "伊", "ロ": "罗"}


def test_effective_glossary_ignores_blank_entries_on_both_sides():
    """手写表里一条只填了键的条目不该把累积的那个译名抹成空串。"""
    effective = g.effective_glossary({"リディア": "莉迪亚"}, {"リディア": "  ", "": "x"})
    assert effective == {"リディア": "莉迪亚"}


def test_effective_glossary_strips_whitespace():
    effective = g.effective_glossary({" イ ": " 伊 "}, {" ロ ": " 罗 "})
    assert effective == {"イ": "伊", "ロ": "罗"}


def test_effective_glossary_does_not_mutate_its_inputs():
    accumulated = {"イ": "伊"}
    manual = {"イ": "壹"}
    g.effective_glossary(accumulated, manual)
    assert accumulated == {"イ": "伊"}
    assert manual == {"イ": "壹"}


def test_effective_glossary_of_nothing_is_empty():
    assert g.effective_glossary({}, {}) == {}


def test_a_saved_table_is_a_fixed_point_of_merge(tmp_path):
    """存盘 → 读回 → 跟空表合并，必须原样回来。

    这三个函数是靠这条不变量串起来的：下一集的累积表是 `merge(load(...), 本集新词)`，
    所以「load 出来的东西恰好是 merge 的不动点」才保证得了译名不会在集与集之间漂。
    """
    path = tmp_path / "glossary.json"
    table = {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}
    g.save_glossary(path, table)
    assert g.merge_glossary(g.load_glossary(path), {}) == table


def test_saving_the_same_table_twice_does_not_touch_the_file(tmp_path):
    """累积表是 script 阶段的新鲜度输入，无条件回写会让每一集的 translate 把前几集
    已经写好的解说稿判旧、下次运行白重跑一遍。"""
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚"})
    os.utime(path, (1_000_000.0, 1_000_000.0))

    g.save_glossary(path, {"リディア": "莉迪亚"})

    assert path.stat().st_mtime == 1_000_000.0


def test_saving_a_changed_table_does_rewrite(tmp_path):
    """上面那条的反面：真有新词时必须落盘，否则跳过写入就变成了丢数据。"""
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚"})
    os.utime(path, (1_000_000.0, 1_000_000.0))

    g.save_glossary(path, {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"})

    assert g.load_glossary(path) == {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}
    assert path.stat().st_mtime != 1_000_000.0


def test_saving_over_a_corrupt_file_still_writes(tmp_path):
    """读盘比对失败（文件坏了 / 不是 UTF-8）时必须退回「照写」，而不是当成「没变」。"""
    path = tmp_path / "glossary.json"
    path.write_bytes(b"\xff\xfe not json")
    os.utime(path, (1_000_000.0, 1_000_000.0))

    g.save_glossary(path, {"リディア": "莉迪亚"})

    assert g.load_glossary(path) == {"リディア": "莉迪亚"}
