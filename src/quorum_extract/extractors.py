"""Extractor protocol, the offline :class:`FakeExtractor`, and provider helpers.

An *extractor* is any callable ``doc -> dict`` (the :class:`Extractor` protocol).
:func:`run_extractors` invokes a list of :class:`~quorum_extract.types.ExtractorSpec`
against one document, capturing failures as ``ok=False`` outputs so K is
preserved (a failed extractor becomes a ``missing`` vote, never a smaller pool).

:class:`FakeExtractor` is the workhorse for deterministic, offline tests and
examples: it returns canned, per-document output and can be configured to agree
or disagree on specific fields, simulate failures, and report a per-invocation
cost.

Provider helpers (OpenAI / Anthropic / Ollama) are **import-guarded**: importing
this module never requires those SDKs; constructing a provider extractor without
its SDK raises a clear, actionable error (M3 surface, designed for offline CI).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .quorum import ExtractorOutput
from .types import DocInput, ExtractorSpec


@runtime_checkable
class Extractor(Protocol):
    """Anything callable that maps a document to a flat-ish dict of values."""

    def __call__(self, doc: DocInput) -> dict[str, Any]: ...


class ExtractorFailure(RuntimeError):
    """Raised inside an extractor to signal a failed/timed-out invocation.

    The orchestrator converts this into an ``ok=False`` output (all-``missing``
    votes) rather than propagating it, so one flaky extractor never aborts a run.
    """


def run_extractors(specs: Sequence[ExtractorSpec], doc: DocInput) -> list[ExtractorOutput]:
    """Invoke each spec against ``doc``, capturing failures as ``ok=False``.

    Any exception raised by an extractor (not only :class:`ExtractorFailure`) is
    caught and recorded as a failed invocation; K is preserved.
    """
    outputs: list[ExtractorOutput] = []
    for spec in specs:
        try:
            data = spec.fn(doc)
            if not isinstance(data, Mapping):
                raise ExtractorFailure(
                    f"extractor {spec.name!r} returned {type(data).__name__}, expected a dict"
                )
            outputs.append(ExtractorOutput(spec=spec, data=dict(data), ok=True))
        except Exception:
            outputs.append(ExtractorOutput(spec=spec, data={}, ok=False))
    return outputs


class FakeExtractor:
    """Deterministic offline extractor for tests, examples, and demos.

    Two construction styles:

    1. Per-document outputs: ``FakeExtractor(name, outputs={doc_id: {...}})`` --
       returns the canned dict for the doc's id (the doc is expected to be a
       mapping carrying an ``id`` key, or a hashable used directly).
    2. A function: ``FakeExtractor(name, fn=lambda doc: {...})``.

    ``fail_on`` is a set of doc ids for which the extractor raises
    :class:`ExtractorFailure` (to exercise the missing-vote path). ``cost_usd``
    and ``tier`` mirror :class:`ExtractorSpec`.
    """

    def __init__(
        self,
        name: str,
        *,
        outputs: Mapping[str, Mapping[str, Any]] | None = None,
        fn: Callable[[DocInput], dict[str, Any]] | None = None,
        cost_usd: float = 0.0,
        tier: int = 0,
        fail_on: set[str] | None = None,
    ) -> None:
        if (outputs is None) == (fn is None):
            raise ValueError("FakeExtractor requires exactly one of `outputs` or `fn`")
        self.name = name
        self._outputs = outputs
        self._fn = fn
        self.cost_usd = cost_usd
        self.tier = tier
        self._fail_on = fail_on or set()
        self.call_count = 0
        self.called_doc_ids: list[str] = []

    @staticmethod
    def doc_id(doc: DocInput) -> str:
        """Best-effort stable id for a document (``doc['id']`` or ``str(doc)``)."""
        if isinstance(doc, Mapping) and "id" in doc:
            return str(doc["id"])
        return str(doc)

    def __call__(self, doc: DocInput) -> dict[str, Any]:
        did = self.doc_id(doc)
        self.call_count += 1
        self.called_doc_ids.append(did)
        if did in self._fail_on:
            raise ExtractorFailure(f"{self.name} configured to fail on {did!r}")
        if self._fn is not None:
            return self._fn(doc)
        assert self._outputs is not None
        if did not in self._outputs:
            raise ExtractorFailure(f"{self.name} has no canned output for {did!r}")
        return dict(self._outputs[did])

    def to_spec(self) -> ExtractorSpec:
        """Wrap this fake as an :class:`ExtractorSpec`."""
        return ExtractorSpec(name=self.name, fn=self, cost_usd=self.cost_usd, tier=self.tier)


# --------------------------------------------------------------------------- #
# Provider helpers (import-guarded; never required to import this module).
# --------------------------------------------------------------------------- #


def _require(module_name: str, extra: str) -> Any:
    """Import an optional provider SDK or raise an actionable error."""
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            f"the {module_name!r} package is required for this provider. "
            f"Install it with: pip install 'quorum-extract[{extra}]'"
        ) from exc


def openai_extractor(
    name: str,
    *,
    model: str,
    schema: type[Any],
    cost_usd: float,
    tier: int = 1,
    client: Any | None = None,
    system_prompt: str | None = None,
) -> ExtractorSpec:
    """Build an :class:`ExtractorSpec` backed by the OpenAI structured-output API.

    Import-guarded: requires ``openai`` only when called. The returned callable
    expects the document to be text (or to expose ``doc['text']``) and uses the
    model's JSON/structured-output mode to fill ``schema``.
    """
    openai = client if client is not None else _require("openai", "openai").OpenAI()

    def _fn(doc: DocInput) -> dict[str, Any]:
        text = doc["text"] if isinstance(doc, Mapping) and "text" in doc else str(doc)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        completion = openai.beta.chat.completions.parse(
            model=model, messages=messages, response_format=schema
        )
        parsed = completion.choices[0].message.parsed
        return dict(parsed.model_dump()) if parsed is not None else {}

    return ExtractorSpec(name=name, fn=_fn, cost_usd=cost_usd, tier=tier)


def anthropic_extractor(
    name: str,
    *,
    model: str,
    cost_usd: float,
    tier: int = 1,
    client: Any | None = None,
    tool_name: str = "extract",
    input_schema: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
) -> ExtractorSpec:
    """Build an :class:`ExtractorSpec` backed by Anthropic tool-use extraction.

    Import-guarded: requires ``anthropic`` only when called. Uses a single
    forced tool call whose ``input_schema`` is the JSON schema of the target
    Pydantic model (pass ``Model.model_json_schema()``).
    """
    anthropic = client if client is not None else _require("anthropic", "anthropic").Anthropic()
    schema = dict(input_schema or {"type": "object"})

    def _fn(doc: DocInput) -> dict[str, Any]:
        text = doc["text"] if isinstance(doc, Mapping) and "text" in doc else str(doc)
        message = anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "Extract the requested fields.",
            tools=[{"name": tool_name, "description": "Extract fields", "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": text}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        return {}

    return ExtractorSpec(name=name, fn=_fn, cost_usd=cost_usd, tier=tier)


def ollama_extractor(
    name: str,
    *,
    model: str,
    schema: Mapping[str, Any],
    cost_usd: float = 0.0,
    tier: int = 0,
    host: str = "http://localhost:11434",
    system_prompt: str | None = None,
) -> ExtractorSpec:
    """Build an :class:`ExtractorSpec` backed by a local Ollama model.

    Import-guarded: requires ``ollama`` only when called. Local models are
    typically ``cost_usd=0`` and ``tier=0`` (cheap baseline). Uses Ollama's
    structured-output ``format`` parameter with the model's JSON schema.
    """
    ollama = _require("ollama", "ollama")
    json_schema = dict(schema)
    ollama_client = ollama.Client(host=host)

    def _fn(doc: DocInput) -> dict[str, Any]:
        import json as _json

        text = doc["text"] if isinstance(doc, Mapping) and "text" in doc else str(doc)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        response = ollama_client.chat(model=model, messages=messages, format=json_schema)
        content = response["message"]["content"]
        return dict(_json.loads(content))

    return ExtractorSpec(name=name, fn=_fn, cost_usd=cost_usd, tier=tier)
