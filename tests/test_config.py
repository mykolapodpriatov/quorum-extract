"""Tests for project config loading and RecordResult JSONL (de)serialization."""

from __future__ import annotations

import pytest

from quorum_extract import (
    EscalationStatus,
    FieldResult,
    FieldVote,
    ProjectConfig,
    RecordResult,
    load_config,
    read_results,
    write_results,
)
from quorum_extract.config import read_labeled, record_from_dict, record_to_dict

from ._helpers import Flat, make_spec

CONFIG_SRC = """
from pydantic import BaseModel
from quorum_extract import ProjectConfig, QuorumConfig, FakeExtractor

class M(BaseModel):
    a: str
    b: int

e1 = FakeExtractor("e1", outputs={"d1": {"a": "x", "b": 1}})
e2 = FakeExtractor("e2", outputs={"d1": {"a": "x", "b": 1}})

config = ProjectConfig(
    schema=M,
    extractors=[e1.to_spec(), e2.to_spec()],
    quorum=QuorumConfig(min_agreement=0.5),
)
"""


def test_load_config_module(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg_path = tmp_path / "project.py"
    cfg_path.write_text(CONFIG_SRC, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, ProjectConfig)
    assert len(cfg.extractors) == 2
    assert cfg.leaf_path_strings() == ["a", "b"]
    assert cfg.extractor_names() == ["e1", "e2"]


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.py")


def test_load_config_resolves_nested_forward_refs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A config using `from __future__ import annotations` with a nested model:
    # the loader must register the module so Pydantic resolves the forward ref,
    # and leaf_paths must therefore expand the nested object (regression test).
    src = (
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "from quorum_extract import ProjectConfig, QuorumConfig, FakeExtractor\n"
        "class Addr(BaseModel):\n"
        "    city: str\n"
        "    zip: str\n"
        "class Doc(BaseModel):\n"
        "    name: str\n"
        "    addr: Addr\n"
        "e = FakeExtractor('e', outputs={'d1': {'name': 'x', 'addr': {'city': 'NYC', 'zip': '1'}}})\n"
        "config = ProjectConfig(schema=Doc, extractors=[e.to_spec()])\n"
    )
    p = tmp_path / "nested.py"
    p.write_text(src, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.leaf_path_strings() == ["addr.city", "addr.zip", "name"]


def test_load_config_exec_error_cleans_up_module(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import sys

    p = tmp_path / "boom.py"
    p.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="boom"):
        load_config(p)
    # The partially-loaded module must not leak into sys.modules.
    leaked = [m for m in set(sys.modules) - before if "user_config" in m]
    assert leaked == []


def test_load_config_no_config_object(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define"):
        load_config(p)


def test_load_config_wrong_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "bad.py"
    p.write_text("config = 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a ProjectConfig"):
        load_config(p)


def test_project_config_requires_extractors() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ProjectConfig(schema=Flat, extractors=[])


def test_project_config_includes_strong_in_names() -> None:
    cfg = ProjectConfig(
        schema=Flat,
        extractors=[make_spec("e1")],
        strong_extractor=make_spec("strong", tier=1),
    )
    assert cfg.extractor_names() == ["e1", "strong"]


def sample_record() -> RecordResult:
    fr = FieldResult(
        path="a",
        value="x",
        votes=[
            FieldVote(extractor="e1", raw_value="x", normalized_key="str:x", missing=False),
            FieldVote(extractor="e2", raw_value=None, normalized_key="__missing__", missing=True),
        ],
        agreement=0.5,
        confidence=0.7,
        status=EscalationStatus.ESCALATED_MODEL,
        winning_key="str:x",
    )
    return RecordResult(doc_id="d1", fields={"a": fr}, cost_usd=0.012)


def test_record_dict_roundtrip() -> None:
    rec = sample_record()
    restored = record_from_dict(record_to_dict(rec))
    assert restored.doc_id == rec.doc_id
    assert restored.cost_usd == rec.cost_usd
    fr = restored.fields["a"]
    assert fr.value == "x"
    assert fr.confidence == 0.7
    assert fr.status is EscalationStatus.ESCALATED_MODEL
    assert len(fr.votes) == 2
    assert fr.votes[1].missing is True


def test_write_read_results_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "results.jsonl"
    records = [sample_record(), sample_record()]
    write_results(path, records)
    loaded = read_results(path)
    assert len(loaded) == 2
    assert loaded[0].fields["a"].value == "x"


def test_read_labeled_parses_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "labels.jsonl"
    p.write_text(
        '{"winning_share": 1.0, "k": 4, "entropy": 0.0, "correct": true}\n'
        '{"winning_share": 0.5, "k": 4, "entropy": 0.69, "correct": false, "group": "g1"}\n',
        encoding="utf-8",
    )
    rows = read_labeled(p)
    assert len(rows) == 2
    assert rows[0].correct is True
    assert rows[0].features.winning_share == 1.0
    assert rows[1].group == "g1"
    assert rows[1].correct is False


def test_read_labeled_missing_keys_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "labels.jsonl"
    p.write_text('{"k": 4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="winning_share"):
        read_labeled(p)
