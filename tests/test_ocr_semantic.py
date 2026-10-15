import json

from tenmin.ingest.srt_parser import load_srt

from .test_ocr_correction import Provider, _paths, _results, _select


class TwoRound(Provider):
    async def complete(self, system, user, schema):
        self.calls.append(json.loads(user))
        if len(self.calls) == 1:
            return schema.model_validate({"lines": _results()})
        return schema.model_validate({"lines": [
            {"id": 1, "text": "垃圾也没有丢", "uncertain": False, "reason": "语境明确"},
            {"id": 2, "text": "汇兑？", "uncertain": True, "reason": "术语有歧义"},
        ]})


def test_semantic_round_sees_original_and_first_pass_and_records_candidate(tmp_path):
    paths = _paths(tmp_path)
    provider = TwoRound()
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[1] and not warnings and not retry
    assert [c.text for c in load_srt(chosen)] == ["垃圾也没有丢", "汇兕"]
    assert len(provider.calls) == 2
    assert provider.calls[1]["correct_these"][0]["original"] == "垃圾也杀丢"
    assert provider.calls[1]["correct_these"][0]["text"] == "垃圾也没丢"
    item = json.loads(paths[3].read_text())[0]
    assert item["candidate"] == "汇兑？" and item["reason"] == "术语有歧义"


def test_semantic_failure_delivers_first_round_and_retries(tmp_path):
    paths = _paths(tmp_path)

    class FailedSecond(TwoRound):
        async def complete(self, system, user, schema):
            if self.calls:
                raise RuntimeError("语义轮失败")
            return await super().complete(system, user, schema)

    chosen, warnings, retry = _select(paths, FailedSecond)
    assert chosen == paths[1] and retry and "语义" in warnings[0]
    assert load_srt(chosen)[0].text == "垃圾也没丢"
    assert json.loads(paths[2].read_text())["status"] == "semantic_pending"
    assert _select(paths, TwoRound) == (paths[1], [], False)


def test_failed_later_semantic_batch_keeps_only_that_batch_first_round(tmp_path):
    from tenmin.ingest.asr import render_srt
    from tenmin.models import RawCue

    paths = _paths(tmp_path)
    paths[0].write_text(render_srt([
        RawCue(idx=i + 1, start=i * 2, end=i * 2 + 1, text="原稿") for i in range(65)
    ]))

    class LaterFailure(Provider):
        async def complete(self, system, user, schema):
            request = json.loads(user)
            self.calls.append(request)
            rows = request["correct_these"]
            semantic = "original" in rows[0]
            if semantic and rows[0]["id"] > 60:
                return schema.model_validate({"lines": []})
            return schema.model_validate({"lines": [
                {"id": row["id"], "text": "二轮" if semantic else "首轮", "uncertain": False}
                for row in rows
            ]})

    chosen, warnings, retry = _select(paths, LaterFailure)
    assert chosen == paths[1] and retry and warnings
    assert [cue.text for cue in load_srt(chosen)] == ["二轮"] * 60 + ["首轮"] * 5
    assert json.loads(paths[2].read_text())["status"] == "partial"


def test_semantic_alignment_error_is_sent_back_for_bounded_retry(tmp_path):
    paths = _paths(tmp_path)

    class MissingThenFixed(TwoRound):
        async def complete(self, system, user, schema):
            if len(self.calls) == 1:
                self.calls.append(json.loads(user))
                return schema.model_validate({"lines": []})
            return await super().complete(system, user, schema)

    provider = MissingThenFixed()
    chosen, warnings, retry = _select(paths, lambda: provider)
    assert chosen == paths[1] and not warnings and not retry
    assert len(provider.calls) == 3
    assert "1" in provider.calls[2]["alignment_error"]
    assert "2" in provider.calls[2]["alignment_error"]


def test_semantic_cannot_remove_explicit_age(tmp_path):
    from tenmin.ingest.asr import render_srt
    from tenmin.models import RawCue

    paths = _paths(tmp_path)
    paths[0].write_text(render_srt([RawCue(idx=1, start=1, end=2, text="一直学到三十岁")]))

    class ChangeAge(Provider):
        async def complete(self, system, user, schema):
            self.calls.append(user)
            text = "一直学到三十岁" if len(self.calls) == 1 else "一直学到能开户为止"
            return schema.model_validate({"lines": [
                {"id": 1, "text": text, "uncertain": False, "reason": "符合剧情"}
            ]})

    chosen, warnings, retry = _select(paths, ChangeAge)
    assert not warnings and not retry
    assert load_srt(chosen)[0].text == "一直学到三十岁"
    assert json.loads(paths[3].read_text())[0]["candidate"] == "一直学到能开户为止"
