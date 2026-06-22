"""Tests for type-aware normalization and the unified ``missing`` bucket."""

from __future__ import annotations

import datetime as dt

import pytest

from quorum_extract import MISSING_KEY, is_missing, normalize_value
from quorum_extract.normalize import ABSENT
from quorum_extract.schema import LeafKind, LeafPath


def scalar(annotation: object = str) -> LeafPath:
    return LeafPath(path="x", kind=LeafKind.SCALAR, annotation=annotation)


def scalar_list(annotation: object = str) -> LeafPath:
    return LeafPath(path="x", kind=LeafKind.SCALAR_LIST, annotation=annotation)


def structural() -> LeafPath:
    return LeafPath(path="x", kind=LeafKind.STRUCTURAL)


# --------------------------------------------------------------------------- #
# The unified missing bucket -- the single most important rule.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [ABSENT, None, "", "   ", "\t\n  "])
def test_missing_bucket_unifies_all_absence_forms(value: object) -> None:
    """absent key / None / "" / whitespace-only ALL map to the same key."""
    assert normalize_value(scalar(), value) == MISSING_KEY
    assert is_missing(value)


def test_missing_distinct_from_real_value() -> None:
    real = normalize_value(scalar(), "hello")
    assert real != MISSING_KEY
    # A literal "missing"-looking string is still a real value, not the bucket.
    assert normalize_value(scalar(), "missing") != MISSING_KEY


def test_missing_is_consistent_across_calls() -> None:
    assert normalize_value(scalar(), None) == normalize_value(scalar(), "  ")


# --------------------------------------------------------------------------- #
# Number tolerance
# --------------------------------------------------------------------------- #


def test_int_float_string_numbers_agree() -> None:
    k = normalize_value(scalar(float), 3)
    assert k == normalize_value(scalar(float), 3.0)
    assert k == normalize_value(scalar(float), "3.000")
    assert k == normalize_value(scalar(float), "3")


def test_near_equal_floats_collapse() -> None:
    a = normalize_value(scalar(float), 1.0000000001)
    b = normalize_value(scalar(float), 1.0000000002)
    assert a == b


def test_distinct_numbers_differ() -> None:
    assert normalize_value(scalar(float), 3.0) != normalize_value(scalar(float), 3.5)


def test_number_with_thousands_separator() -> None:
    assert normalize_value(scalar(float), "1,000") == normalize_value(scalar(float), 1000)


def test_negative_zero_collapses() -> None:
    assert normalize_value(scalar(float), -0.0) == normalize_value(scalar(float), 0.0)


# --------------------------------------------------------------------------- #
# Dates as instants
# --------------------------------------------------------------------------- #


def test_date_formats_agree_as_instant() -> None:
    k = normalize_value(scalar(dt.date), "2020-01-02")
    assert k == normalize_value(scalar(dt.date), "Jan 2, 2020")
    assert k == normalize_value(scalar(dt.date), dt.date(2020, 1, 2))


def test_datetime_instant_equivalence_across_tz() -> None:
    a = normalize_value(scalar(dt.datetime), "2020-01-02T12:00:00+00:00")
    b = normalize_value(scalar(dt.datetime), "2020-01-02T13:00:00+01:00")
    assert a == b


def test_distinct_dates_differ() -> None:
    assert normalize_value(scalar(dt.date), "2020-01-02") != normalize_value(
        scalar(dt.date), "2020-01-03"
    )


# --------------------------------------------------------------------------- #
# Strings: case + whitespace
# --------------------------------------------------------------------------- #


def test_case_and_whitespace_insensitive_strings() -> None:
    k = normalize_value(scalar(str), "ACME Corp")
    assert k == normalize_value(scalar(str), "acme corp")
    assert k == normalize_value(scalar(str), "  ACME   Corp  ")
    assert k == normalize_value(scalar(str), "acme\tcorp")


def test_distinct_strings_differ() -> None:
    assert normalize_value(scalar(str), "acme") != normalize_value(scalar(str), "globex")


# --------------------------------------------------------------------------- #
# Scalar lists: order-insensitive
# --------------------------------------------------------------------------- #


def test_scalar_list_order_insensitive() -> None:
    k = normalize_value(scalar_list(str), ["a", "b", "c"])
    assert k == normalize_value(scalar_list(str), ["c", "a", "b"])
    assert k == normalize_value(scalar_list(str), {"b", "c", "a"})


def test_scalar_list_distinct_contents_differ() -> None:
    assert normalize_value(scalar_list(str), ["a", "b"]) != normalize_value(
        scalar_list(str), ["a", "c"]
    )


def test_scalar_list_numbers_normalized_per_element() -> None:
    assert normalize_value(scalar_list(float), [1, 2, 3]) == normalize_value(
        scalar_list(float), [3.0, "2", 1.0]
    )


def test_empty_scalar_list_is_missing() -> None:
    # An empty list is *not* whitespace; it is a real (empty) structure, but the
    # is_missing rule only covers absence forms, so it is a real distinct key.
    k = normalize_value(scalar_list(str), [])
    assert k != MISSING_KEY


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #


def test_structural_key_order_insensitive() -> None:
    a = normalize_value(structural(), {"x": 1, "y": 2})
    b = normalize_value(structural(), {"y": 2, "x": 1})
    assert a == b


def test_structural_distinct_differs() -> None:
    assert normalize_value(structural(), {"x": 1}) != normalize_value(structural(), {"x": 2})


def test_bool_distinct_from_int() -> None:
    # True must not collapse into the number 1's bucket.
    assert normalize_value(scalar(int), True) != normalize_value(scalar(int), 1)


# --------------------------------------------------------------------------- #
# Edge cases: special floats, enums, structural recursion, object-list leaves.
# --------------------------------------------------------------------------- #


def test_nan_and_inf_numbers() -> None:
    nan_a = normalize_value(scalar(float), float("nan"))
    nan_b = normalize_value(scalar(float), float("nan"))
    assert nan_a == nan_b == "num:nan"
    assert normalize_value(scalar(float), float("inf")) == "num:inf"
    assert normalize_value(scalar(float), float("-inf")) == "num:-inf"


def test_decimal_value() -> None:
    from decimal import Decimal

    assert normalize_value(scalar(float), Decimal("3.0")) == normalize_value(scalar(float), 3)


def test_enum_value_canonicalized() -> None:
    import enum

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    # Enums fall through to the enum branch (no scalar annotation match).
    k_red = normalize_value(scalar(object), Color.RED)
    assert k_red.startswith("enum:")
    assert k_red != normalize_value(scalar(object), Color.BLUE)


def test_non_numeric_string_stays_string() -> None:
    assert normalize_value(scalar(str), "hello world").startswith("str:")


def test_unparseable_date_string_falls_back_to_string() -> None:
    # A string with a date annotation that cannot be parsed -> string key.
    k = normalize_value(scalar(str), "not a date at all zzz")
    assert k.startswith("str:")


def test_structural_with_nested_list_and_float() -> None:
    a = normalize_value(structural(), {"items": [1.0, 2.0], "n": 3.0})
    b = normalize_value(structural(), {"n": 3, "items": [1.0, 2.0]})
    assert a == b


def test_structural_set_is_order_insensitive() -> None:
    a = normalize_value(structural(), {"s": {3, 1, 2}})
    b = normalize_value(structural(), {"s": {2, 3, 1}})
    assert a == b


def test_object_list_field_uses_scalar_canon() -> None:
    leaf = LeafPath(
        path="items[*].qty",
        kind=LeafKind.OBJECT_LIST_FIELD,
        annotation=int,
        list_field="items",
        list_key="sku",
    )
    assert normalize_value(leaf, "5") == normalize_value(leaf, 5)


def test_object_list_field_nested_dict_canon_structural() -> None:
    # A keyed-row sub-value that is a dict must canonicalize structurally
    # (sorted keys, recursive) -- not via scalar repr, which splits by key order.
    leaf = LeafPath(
        path="items[*].meta",
        kind=LeafKind.OBJECT_LIST_FIELD,
        annotation=dict,
        list_field="items",
        list_key="sku",
    )
    a = normalize_value(leaf, {"x": 1, "y": 2})
    b = normalize_value(leaf, {"y": 2, "x": 1})
    assert a == b
    assert a.startswith("struct:")
    # Distinct dicts still differ.
    assert normalize_value(leaf, {"x": 1}) != normalize_value(leaf, {"x": 2})


def test_object_list_field_nested_list_canon_structural() -> None:
    # A keyed-row sub-value that is a list of dicts canonicalizes structurally.
    leaf = LeafPath(
        path="items[*].tags",
        kind=LeafKind.OBJECT_LIST_FIELD,
        annotation=None,
        list_field="items",
        list_key="sku",
    )
    a = normalize_value(leaf, [{"x": 1, "y": 2}])
    b = normalize_value(leaf, [{"y": 2, "x": 1}])
    assert a == b
    assert a.startswith("struct:")


def _objlist_leaf() -> LeafPath:
    return LeafPath(
        path="items[*].meta",
        kind=LeafKind.OBJECT_LIST_FIELD,
        annotation=dict,
        list_field="items",
        list_key="sku",
    )


def test_structural_nested_string_case_and_whitespace_normalized() -> None:
    # The structural canonicalizer must run nested SCALAR strings through the same
    # case/whitespace folding as top-level scalars, so {"name":"ACME"} and
    # {"name":"acme "} AGREE. Regression: raw repr() split them by formatting.
    leaf = _objlist_leaf()
    assert normalize_value(leaf, {"name": "ACME"}) == normalize_value(leaf, {"name": "acme "})
    # A genuinely different nested string must still DISAGREE.
    assert normalize_value(leaf, {"name": "ACME"}) != normalize_value(leaf, {"name": "GLOBEX"})


def test_structural_nested_number_within_tolerance_agrees() -> None:
    # A nested number within float tolerance (and int/float equivalence) agrees;
    # a genuinely different nested number still differs.
    leaf = _objlist_leaf()
    assert normalize_value(leaf, {"n": 1.0000000001}) == normalize_value(leaf, {"n": 1.0000000002})
    assert normalize_value(leaf, {"n": 3}) == normalize_value(leaf, {"n": 3.0})
    assert normalize_value(leaf, {"n": 3.0}) != normalize_value(leaf, {"n": 3.5})


def test_structural_nested_date_as_instant_agrees() -> None:
    # A nested date in two textual formats representing the same instant agrees;
    # a genuinely different nested date still differs.
    leaf = _objlist_leaf()
    assert normalize_value(leaf, {"d": "2020-01-02"}) == normalize_value(leaf, {"d": "Jan 2, 2020"})
    assert normalize_value(leaf, {"t": "2020-01-02T12:00:00+00:00"}) == normalize_value(
        leaf, {"t": "2020-01-02T13:00:00+01:00"}
    )
    assert normalize_value(leaf, {"d": "2020-01-02"}) != normalize_value(leaf, {"d": "2020-01-03"})


def test_structural_nested_scalars_in_list_normalized() -> None:
    # Nested scalars inside a list (preserving list order) are normalized too:
    # case/whitespace-folded strings inside a list agree.
    leaf = _objlist_leaf()
    a = normalize_value(leaf, {"names": ["ACME", "Globex"]})
    b = normalize_value(leaf, {"names": ["acme ", " globex"]})
    assert a == b
    # List ORDER is still significant (these are not sets).
    assert normalize_value(leaf, {"names": ["a", "b"]}) != normalize_value(
        leaf, {"names": ["b", "a"]}
    )


def test_structural_nested_bool_stays_distinct_from_int() -> None:
    # Routing nested scalars through the scalar normalizer must NOT let a nested
    # True collapse into the number 1's bucket.
    leaf = _objlist_leaf()
    assert normalize_value(leaf, {"flag": True}) != normalize_value(leaf, {"flag": 1})


def test_top_level_structural_nested_string_normalized() -> None:
    # The same recursion applies to a top-level STRUCTURAL (dict) leaf.
    a = normalize_value(structural(), {"name": "ACME Corp"})
    b = normalize_value(structural(), {"name": "  acme   corp "})
    assert a == b
    assert normalize_value(structural(), {"name": "ACME"}) != normalize_value(
        structural(), {"name": "OTHER"}
    )


def test_scalar_list_non_sequence_falls_back_structural() -> None:
    # Passing a non-sequence to a scalar-list leaf -> structural fallback (no crash).
    k = normalize_value(scalar_list(int), 42)
    assert k.startswith("struct:")


def test_int_string_with_int_annotation() -> None:
    assert normalize_value(scalar(int), "42") == normalize_value(scalar(int), 42)


def test_non_scalar_object_repr_fallback() -> None:
    # An object that is neither number, date, string, nor enum -> repr key.
    class Thing:
        def __repr__(self) -> str:
            return "Thing()"

    k = normalize_value(scalar(object), Thing())
    assert k.startswith("repr:")


def test_structural_repr_fallback_for_objects() -> None:
    class Thing:
        def __repr__(self) -> str:
            return "T"

    a = normalize_value(structural(), {"obj": Thing()})
    b = normalize_value(structural(), {"obj": Thing()})
    assert a == b


def test_structural_bool_distinct_from_int() -> None:
    a = normalize_value(structural(), {"flag": True})
    b = normalize_value(structural(), {"flag": 1})
    assert a != b


def test_date_object_passed_to_string_branch() -> None:
    import datetime as _dt

    # A real date with no annotation hint still canonicalizes as a date instant.
    k = normalize_value(scalar(object), _dt.date(2021, 5, 1))
    assert k.startswith("date:")


def test_datetime_with_time_component() -> None:
    import datetime as _dt

    k = normalize_value(scalar(_dt.datetime), _dt.datetime(2021, 5, 1, 14, 30, 0))
    assert k.startswith("datetime:")


def test_string_that_is_a_date_without_annotation() -> None:
    # A date-looking string with a plain str annotation is detected as a date.
    k = normalize_value(scalar(str), "2020-12-25")
    assert k.startswith("date:")
