"""Tests for leaf-path enumeration (the votable grammar)."""

from __future__ import annotations

import datetime as dt
from typing import Optional, Union

import pytest
from pydantic import BaseModel

from quorum_extract import LeafKind, SchemaError, leaf_path_strings, leaf_paths


class Address(BaseModel):
    city: str
    zip: str


class LineItem(BaseModel):
    sku: str
    qty: int


class Doc(BaseModel):
    vendor: str
    total: float
    address: Address
    tags: list[str] = []
    line_items: list[LineItem] = []


def test_scalar_and_nested_paths() -> None:
    paths = leaf_path_strings(Doc)
    # Nested object expands with a dotted prefix; scalar list is one leaf;
    # un-keyed object list is one structural leaf.
    assert paths == [
        "address.city",
        "address.zip",
        "line_items",
        "tags",
        "total",
        "vendor",
    ]


def test_scalar_list_is_one_leaf() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Doc)}
    assert leaves["tags"].kind is LeafKind.SCALAR_LIST
    assert leaves["tags"].annotation is str


def test_object_list_without_key_is_structural() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Doc)}
    assert leaves["line_items"].kind is LeafKind.STRUCTURAL


def test_object_list_with_key_expands() -> None:
    paths = leaf_path_strings(Doc, list_key={"line_items": "sku"})
    assert "line_items[*].sku" in paths
    assert "line_items[*].qty" in paths
    # The bare list path is no longer present once expanded.
    assert "line_items" not in paths


def test_object_list_field_metadata() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Doc, list_key={"line_items": "sku"})}
    qty = leaves["line_items[*].qty"]
    assert qty.kind is LeafKind.OBJECT_LIST_FIELD
    assert qty.list_field == "line_items"
    assert qty.list_key == "sku"
    assert qty.annotation is int


def test_unknown_list_key_raises() -> None:
    with pytest.raises(SchemaError, match="references sub-field"):
        leaf_paths(Doc, list_key={"line_items": "nonexistent"})


def test_optional_nested_object_still_contributes_leaves() -> None:
    class WithOptional(BaseModel):
        name: str
        address: Optional[Address] = None  # noqa: UP045 - exercise Optional form

    paths = leaf_path_strings(WithOptional)
    # An optional nested object still contributes its leaves (missing handled
    # at vote time, not enumeration time).
    assert "address.city" in paths
    assert "address.zip" in paths


def test_union_scalar_is_single_leaf() -> None:
    class WithUnion(BaseModel):
        value: Union[int, str]  # noqa: UP007 - exercise Union form

    leaves = {lp.path: lp for lp in leaf_paths(WithUnion)}
    assert leaves["value"].kind is LeafKind.SCALAR


def test_dict_is_structural() -> None:
    class WithDict(BaseModel):
        meta: dict[str, int] = {}

    leaves = {lp.path: lp for lp in leaf_paths(WithDict)}
    assert leaves["meta"].kind is LeafKind.STRUCTURAL


def test_date_annotation_preserved() -> None:
    class WithDate(BaseModel):
        when: dt.date

    leaves = {lp.path: lp for lp in leaf_paths(WithDate)}
    assert leaves["when"].annotation is dt.date


def test_recursive_model_terminates() -> None:
    class Node(BaseModel):
        name: str
        child: Optional[Node] = None  # noqa: UP045

    Node.model_rebuild()
    paths = leaf_path_strings(Node)
    # Recursion stops at the self-reference as a single structural leaf.
    assert "name" in paths
    assert any("child" in p for p in paths)


def test_non_model_raises() -> None:
    with pytest.raises(SchemaError):
        leaf_paths(int)  # type: ignore[arg-type]


def test_paths_are_sorted_deterministic() -> None:
    p1 = leaf_path_strings(Doc)
    p2 = leaf_path_strings(Doc)
    assert p1 == p2 == sorted(p1)


def test_homogeneous_tuple_is_scalar_list() -> None:
    class WithTuple(BaseModel):
        coords: tuple[float, ...] = ()

    leaves = {lp.path: lp for lp in leaf_paths(WithTuple)}
    assert leaves["coords"].kind is LeafKind.SCALAR_LIST


def test_heterogeneous_tuple_is_structural() -> None:
    class WithTuple(BaseModel):
        pair: tuple[int, str] = (0, "")

    leaves = {lp.path: lp for lp in leaf_paths(WithTuple)}
    assert leaves["pair"].kind is LeafKind.STRUCTURAL
