"""OCR 文字校对：逐条对齐、保留不确定文本、缓存与人工修改保护。"""

import asyncio
import json
from pathlib import Path

import pytest

from tenmin.config import LLMConfig
from tenmin.ingest.asr import render_srt
from tenmin.ingest.ocr_correction import correct_cues, select_corrected
from tenmin.ingest.srt_parser import load_srt
from tenmin.models import RawCue


def _cues():
    return [
        RawCue(idx=1, start=1, end=2, text="垃圾也杀丢"),
        RawCue(idx=2, start=3, end=4, text="汇兕"),
    ]


class Provider:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []
        self.closed = False

    async def complete(self, system, user, schema):
        self.calls.append(user)
        if self.error:
            raise self.error
        return schema.model_validate({"lines": self.results})

    async def aclose(self):
        self.closed = True


def _results():
    return [
        {"id": 1, "text": "垃圾也没丢", "uncertain": False, "reason": ""},
        {"id": 2, "text": "汇兑", "uncertain": True, "reason": "仅凭文字无法确认"},
    ]


def test_confident_change_keeps_timing_and_uncertain_line_unchanged():
    provider = Provider(_results())
    corrected, review = asyncio.run(correct_cues(_cues(), provider, cfg=LLMConfig()))
    assert [(c.idx, c.start, c.end, c.text) for c in corrected] == [
        (1, 1, 2, "垃圾也没丢"),
        (2, 3, 4, "汇兕"),
    ]
    assert [(r.id, r.text, r.reason) for r in review] == [(2, "汇兕", "仅凭文字无法确认")]
    assert "垃圾也杀丢" in provider.calls[0]
    assert "汇兕" in provider.calls[0]


@pytest.mark.parametrize("ids", [[1, 1], [1], [1, 3]])
def test_alignment_rejects_duplicates_missing_and_unknown(ids):
    provider = Provider([{"id": i, "text": "好的", "uncertain": False, "reason": ""} for i in ids])
    with pytest.raises(ValueError, match="对齐"):
        asyncio.run(correct_cues(_cues(), provider, cfg=LLMConfig()))


def test_empty_corrected_text_rejects_whole_document():
    result = _results()
    result[0]["text"] = "   "
    with pytest.raises(ValueError, match="空"):
        asyncio.run(correct_cues(_cues(), Provider(result), cfg=LLMConfig()))


def test_text_cannot_insert_extra_srt_blocks():
    result = _results()
    result[0]["text"] = "台词\n\n3\n00:00:05,000 --> 00:00:06,000\n额外字幕"
    with pytest.raises(ValueError, match="文本"):
        asyncio.run(correct_cues(_cues(), Provider(result), cfg=LLMConfig()))


def _paths(tmp_path: Path):
    raw = tmp_path / "E01.ocr.srt"
    raw.write_text(render_srt(_cues()), encoding="utf-8")
    return (
        raw, tmp_path / "E01.ocr.corrected.srt", tmp_path / "state.json", tmp_path / "review.json"
    )


def _select(paths, factory):
    raw, corrected, state, review = paths
    return asyncio.run(
        select_corrected(
            raw,
            corrected,
            state,
            review,
            provider_factory=factory,
            cfg=LLMConfig(),
            source_identity="source-v1",
        )
    )


def test_success_caches_and_preserves_human_edit(tmp_path):
    paths = _paths(tmp_path)
    provider = Provider(_results())
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert (chosen, warnings, retry) == (paths[1], [], False)
    assert provider.closed
    assert [c.text for c in load_srt(paths[1])] == ["垃圾也没丢", "汇兕"]
    assert json.loads(paths[3].read_text())[0]["reason"] == "仅凭文字无法确认"
    assert _select(paths, lambda: pytest.fail("must reuse"))[0] == paths[1]
    paths[1].write_text(render_srt([c.model_copy(update={"text": "手工"}) for c in _cues()]))
    assert _select(paths, lambda: pytest.fail("must preserve human edit"))[0] == paths[1]
    assert load_srt(paths[1])[0].text == "手工"


def test_changed_raw_with_human_edit_preserves_old_and_uses_new_raw(tmp_path):
    paths = _paths(tmp_path)
    _select(paths, lambda: Provider(_results()))
    paths[1].write_text("手工终稿")
    paths[0].write_text(render_srt([c.model_copy(update={"text": "新原稿"}) for c in _cues()]))
    chosen, warnings, retry = _select(paths, lambda: pytest.fail("do not overwrite"))
    assert chosen == paths[0] and warnings and retry
    assert paths[1].read_text() == "手工终稿"


def test_missing_key_or_request_failure_falls_back_and_retries(tmp_path):
    paths = _paths(tmp_path)
    calls = 0

    def missing():
        nonlocal calls
        calls += 1
        raise RuntimeError("缺少 key")

    assert _select(paths, missing) == (paths[0], ["OCR AI 校对失败：缺少 key"], True)
    assert _select(paths, missing)[2] and calls == 2
    provider = Provider(error=RuntimeError("请求超时"))
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[0] and retry and "请求超时" in warnings[0]
    assert provider.closed
    assert not paths[2].exists()


def test_unmanaged_existing_corrected_file_is_never_overwritten(tmp_path):
    paths = _paths(tmp_path)
    paths[1].write_text("既有字幕")
    chosen, warnings, retry = _select(paths, lambda: pytest.fail("no provider"))
    assert chosen == paths[0] and warnings and retry
    assert paths[1].read_text() == "既有字幕"


def test_model_change_preserves_human_edit_when_source_is_same(tmp_path):
    paths = _paths(tmp_path)
    _select(paths, lambda: Provider(_results()))
    paths[1].write_text("人工稿")
    chosen, _warnings, retry = asyncio.run(select_corrected(
        *paths, provider_factory=lambda: pytest.fail("must preserve"),
        cfg=LLMConfig(model="other-model"), source_identity="source-v1",
    ))
    assert chosen == paths[1] and not retry


def test_failed_later_batch_does_not_write_partial_correction(tmp_path):
    paths = _paths(tmp_path)
    cues = [RawCue(idx=i + 1, start=i * 2, end=i * 2 + 1, text="原稿") for i in range(65)]
    paths[0].write_text(render_srt(cues))

    class LaterFailure(Provider):
        async def complete(self, system, user, schema):
            self.calls.append(user)
            if len(self.calls) == 2:
                raise RuntimeError("第二批失败")
            batch = json.loads(user)["correct_these"]
            return schema.model_validate({"lines": [
                {"id": x["id"], "text": "修正", "uncertain": False} for x in batch
            ]})

    provider = LaterFailure()
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[0] and retry and warnings
    assert len(provider.calls) == 2 and not paths[1].exists()


def test_provider_cleanup_failure_warns_and_falls_back(tmp_path):
    paths = _paths(tmp_path)

    class CloseFailure(Provider):
        async def aclose(self):
            raise RuntimeError("关闭连接失败")

    chosen, warnings, retry = _select(paths, lambda: CloseFailure(_results()))
    assert chosen == paths[0] and retry
    assert "关闭连接失败" in warnings[0]
    assert not paths[1].exists()


def test_malformed_raw_is_not_silently_reduced(tmp_path):
    paths = _paths(tmp_path)
    paths[0].write_text(paths[0].read_text() + "\n3\ninvalid timestamp\n不能丢的原文\n")
    provider = Provider(_results())
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[0] and warnings and retry
    assert not provider.calls and not paths[1].exists()


def test_batches_have_read_only_context_and_preserve_original_order():
    cues = [RawCue(idx=i + 1, start=i * 2, end=i * 2 + 1, text=str(i)) for i in range(65)]

    class Echo(Provider):
        async def complete(self, system, user, schema):
            request = json.loads(user)
            self.calls.append(request)
            return schema.model_validate({"lines": [
                {"id": x["id"], "text": x["text"], "uncertain": False}
                for x in reversed(request["correct_these"])
            ]})

    provider = Echo()
    corrected, review = asyncio.run(correct_cues(cues, provider, cfg=LLMConfig()))
    assert corrected == cues and not review
    assert [x["id"] for x in provider.calls[0]["context_only"]] == [61, 62]
    assert [x["id"] for x in provider.calls[1]["context_only"]] == [59, 60]
