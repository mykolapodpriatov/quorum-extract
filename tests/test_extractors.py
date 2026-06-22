"""Tests for the extractor protocol, FakeExtractor, and import-guarded providers."""

from __future__ import annotations

import builtins

import pytest

from quorum_extract import (
    Extractor,
    ExtractorFailure,
    FakeExtractor,
    anthropic_extractor,
    ollama_extractor,
    openai_extractor,
    run_extractors,
)
from quorum_extract.extractors import _require


def test_fake_extractor_returns_canned_output() -> None:
    fake = FakeExtractor("f", outputs={"d1": {"a": "x"}})
    assert fake({"id": "d1"}) == {"a": "x"}
    assert fake.call_count == 1


def test_fake_extractor_fn_style() -> None:
    fake = FakeExtractor("f", fn=lambda doc: {"len": len(doc["text"])})
    assert fake({"id": "d1", "text": "hello"}) == {"len": 5}


def test_fake_extractor_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FakeExtractor("f")
    with pytest.raises(ValueError, match="exactly one"):
        FakeExtractor("f", outputs={}, fn=lambda d: {})


def test_fake_extractor_fail_on() -> None:
    fake = FakeExtractor("f", outputs={"d1": {"a": "x"}}, fail_on={"d1"})
    with pytest.raises(ExtractorFailure):
        fake({"id": "d1"})


def test_fake_extractor_missing_output_raises() -> None:
    fake = FakeExtractor("f", outputs={"d1": {"a": "x"}})
    with pytest.raises(ExtractorFailure, match="no canned output"):
        fake({"id": "d2"})


def test_fake_extractor_doc_id_fallback() -> None:
    assert FakeExtractor.doc_id({"id": "abc"}) == "abc"
    assert FakeExtractor.doc_id("plain") == "plain"


def test_fake_extractor_is_extractor_protocol() -> None:
    fake = FakeExtractor("f", outputs={"d1": {}})
    assert isinstance(fake, Extractor)


def test_run_extractors_captures_failure_preserving_k() -> None:
    good = FakeExtractor("good", outputs={"d1": {"a": "x"}}).to_spec()
    bad = FakeExtractor("bad", outputs={"d1": {"a": "x"}}, fail_on={"d1"}).to_spec()
    outputs = run_extractors([good, bad], {"id": "d1"})
    assert len(outputs) == 2  # K preserved
    assert outputs[0].ok is True
    assert outputs[1].ok is False
    assert outputs[1].data == {}


def test_run_extractors_non_dict_return_is_failure() -> None:
    bad = FakeExtractor("bad", fn=lambda doc: ["not", "a", "dict"]).to_spec()  # type: ignore[arg-type,return-value]
    outputs = run_extractors([bad], {"id": "d1"})
    assert outputs[0].ok is False


def test_to_spec_carries_cost_and_tier() -> None:
    spec = FakeExtractor("f", outputs={"d1": {}}, cost_usd=0.02, tier=2).to_spec()
    assert spec.cost_usd == 0.02
    assert spec.tier == 2
    assert spec.name == "f"


# --------------------------------------------------------------------------- #
# Provider helpers are import-guarded.
# --------------------------------------------------------------------------- #


def test_require_missing_module_raises_actionable() -> None:
    with pytest.raises(ImportError, match="quorum-extract"):
        _require("definitely_not_a_real_module_xyz", "openai")


def test_provider_helpers_guard_when_sdk_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("openai", "anthropic", "ollama"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="openai"):
        openai_extractor("o", model="gpt", schema=dict, cost_usd=0.01)
    with pytest.raises(ImportError, match="anthropic"):
        anthropic_extractor("a", model="claude", cost_usd=0.01)
    with pytest.raises(ImportError, match="ollama"):
        ollama_extractor("ol", model="llama", schema={})


def test_openai_extractor_with_injected_client() -> None:
    """A fake OpenAI-shaped client lets us build the spec offline."""

    class _Msg:
        parsed = type("P", (), {"model_dump": lambda self: {"a": "x"}})()

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _Parse:
        def parse(self, **kwargs: object) -> _Completion:
            return _Completion()

    class _Chat:
        completions = _Parse()

    class _Beta:
        chat = _Chat()

    class _Client:
        beta = _Beta()

    spec = openai_extractor("o", model="gpt", schema=dict, cost_usd=0.01, client=_Client())
    assert spec.name == "o"
    assert spec.fn({"text": "hello"}) == {"a": "x"}


def test_anthropic_extractor_with_injected_client() -> None:
    class _Block:
        type = "tool_use"
        input = {"a": "x"}

    class _Message:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs: object) -> _Message:
            return _Message()

    class _Client:
        messages = _Messages()

    spec = anthropic_extractor("a", model="claude", cost_usd=0.01, client=_Client())
    assert spec.fn({"text": "hello"}) == {"a": "x"}


def test_anthropic_extractor_no_tool_use_returns_empty() -> None:
    class _Block:
        type = "text"

    class _Message:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs: object) -> _Message:
            return _Message()

    class _Client:
        messages = _Messages()

    spec = anthropic_extractor("a", model="claude", cost_usd=0.01, client=_Client())
    assert spec.fn({"text": "hello"}) == {}


def test_ollama_extractor_via_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json
    import types as _types

    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, host: str) -> None:
            captured["host"] = host

        def chat(self, model: str, messages: list, format: object) -> dict:
            captured["model"] = model
            captured["format"] = format
            return {"message": {"content": _json.dumps({"a": "x"})}}

    fake_ollama = _types.SimpleNamespace(Client=_FakeClient)
    monkeypatch.setattr("quorum_extract.extractors._require", lambda name, extra: fake_ollama)

    spec = ollama_extractor("ol", model="llama", schema={"type": "object"}, system_prompt="extract")
    assert spec.fn({"text": "hello world"}) == {"a": "x"}
    assert captured["model"] == "llama"
    assert captured["host"] == "http://localhost:11434"


def test_openai_extractor_none_parsed_returns_empty() -> None:
    class _Msg:
        parsed = None

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _Parse:
        def parse(self, **kwargs: object) -> _Completion:
            return _Completion()

    class _Chat:
        completions = _Parse()

    class _Beta:
        chat = _Chat()

    class _Client:
        beta = _Beta()

    spec = openai_extractor(
        "o", model="gpt", schema=dict, cost_usd=0.01, client=_Client(), system_prompt="extract"
    )
    assert spec.fn("plain text doc") == {}
