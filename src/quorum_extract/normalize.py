"""Type-aware, deterministic value normalization for vote equality.

The crux of quorum extraction (plan 7): naive string equality undercounts
agreement because two correct extractors format the same fact differently
(``"3.0"`` vs ``3``, ``"2020-01-02"`` vs ``"Jan 2, 2020"``, ``"ACME"`` vs
``"acme"``). :func:`normalize_value` maps a raw value to a canonical
``normalized_key`` string; two votes agree iff their keys match.

The single most important rule is the **unified ``missing`` bucket**: an absent
key, ``None``, ``""``, or whitespace-only value ALL map to the same sentinel
key, and that key never equals any real value. (A failed extractor is recorded
as ``missing`` for every field upstream, in :mod:`quorum_extract.quorum`.)

Everything here is deterministic: no RNG, no clock, no set-iteration order
leaking into a key.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum as _Enum
from typing import Any

from dateutil import parser as _date_parser

from .schema import LeafKind, LeafPath

#: The single sentinel key for the unified absence bucket. Chosen to be a string
#: that cannot collide with a normalized real value (which are prefixed by type).
MISSING_KEY = "__missing__"

#: Relative+absolute tolerance for treating two floats as the same vote.
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-9

# A unique object distinct from ``None`` so callers can express "the key was
# absent" separately from "the value was explicitly None". Both still normalize
# to MISSING_KEY -- they are unified by design -- but the distinction lets the
# vote layer pass absence through cleanly.
ABSENT: Any = object()


def is_missing(value: Any) -> bool:
    """True iff ``value`` belongs in the unified ``missing`` bucket.

    Covers: the :data:`ABSENT` sentinel, ``None``, the empty string, and any
    string that is whitespace-only.
    """
    if value is ABSENT or value is None:
        return True
    return bool(isinstance(value, str) and value.strip() == "")


def _canon_number(value: float | int | Decimal) -> str:
    """Canonicalize a real number so that ``3``, ``3.0``, ``"3.000"`` agree.

    Integers and integer-valued floats share a key; non-integers are rounded to
    a fixed precision derived from the float tolerance so near-equal values
    collapse together deterministically.
    """
    f = float(value)
    if math.isnan(f):
        return "num:nan"
    if math.isinf(f):
        return "num:inf" if f > 0 else "num:-inf"
    if f == 0.0:
        f = 0.0  # collapse -0.0
    rounded = round(f, 9)
    if rounded == int(rounded):
        return f"num:{int(rounded)}"
    # Trim trailing zeros for a stable textual key.
    return f"num:{rounded:.9f}".rstrip("0").rstrip(".")


def _coerce_number(value: Any) -> float | None:
    """Best-effort numeric coercion for SCALAR fields annotated as numbers.

    Returns ``None`` if ``value`` is not numeric-like.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | Decimal):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _canon_date(value: Any) -> str | None:
    """Canonicalize dates/datetimes to a single instant key, or ``None``.

    Accepts ``datetime``/``date`` objects and parseable strings. Datetimes are
    compared as instants; naive datetimes are assumed UTC for a stable key.
    """
    dt: _dt.datetime | None = None
    if isinstance(value, _dt.datetime):
        dt = value
    elif isinstance(value, _dt.date):
        return f"date:{value.isoformat()}"
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = _date_parser.parse(s)
        except (ValueError, OverflowError):
            return None
        dt = parsed
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    instant = dt.astimezone(_dt.UTC)
    midnight = instant.hour == 0 and instant.minute == 0
    if midnight and instant.second == 0 and instant.microsecond == 0:
        return f"date:{instant.date().isoformat()}"
    return f"datetime:{instant.isoformat()}"


def _canon_string(value: str) -> str:
    """Case- and whitespace-insensitive string key (internal runs collapsed)."""
    collapsed = " ".join(value.split())
    return f"str:{collapsed.casefold()}"


def _canon_scalar(value: Any, annotation: Any) -> str:
    """Canonicalize a single scalar value using its annotation as a hint."""
    if isinstance(value, bool):
        return f"bool:{value}"

    # Date-like annotation (or value) first, so a date string isn't eaten by the
    # string branch.
    if annotation in (_dt.date, _dt.datetime) or isinstance(value, _dt.date | _dt.datetime):
        date_key = _canon_date(value)
        if date_key is not None:
            return date_key

    # Numeric annotation or numeric-looking value.
    if annotation in (int, float, Decimal) or isinstance(value, int | float | Decimal):
        num = _coerce_number(value)
        if num is not None:
            return _canon_number(num)

    if isinstance(value, str):
        # A numeric or date string with no numeric/date annotation: try to be
        # smart but fall back to a normalized string key.
        num = _coerce_number(value)
        if num is not None:
            return _canon_number(num)
        date_key = _canon_date(value)
        if date_key is not None:
            return date_key
        return _canon_string(value)

    if isinstance(value, _Enum):
        return f"enum:{value.value!r}"

    # Fallback: repr-based structural key (stable for hashable scalars).
    return f"repr:{value!r}"


def _canon_scalar_list(value: Any, element_annotation: Any) -> str:
    """Order-insensitive key for a list/set/tuple of scalars."""
    if not isinstance(value, Sequence | set | frozenset) or isinstance(value, str | bytes):
        # Not a sequence we recognize -> structural fallback.
        return _canon_structural(value)
    items = [_canon_scalar(v, element_annotation) for v in value]
    items.sort()
    return "list:[" + ",".join(items) + "]"


def _canon_structural(value: Any) -> str:
    """Deterministic structural key for dicts / un-keyed lists / nested data.

    Sorts mapping keys and normalizes recursively so equivalent structures with
    different key order collapse to the same key. Nested *scalar* leaves are run
    through the same per-type scalar normalizers used at top level (string
    case/whitespace folding, number tolerance, date-as-instant), so two
    semantically-equal nested payloads collapse to one bucket instead of
    splitting on raw ``repr`` formatting.
    """
    return "struct:" + _structural_repr(value)


def _structural_repr(value: Any) -> str:
    # Booleans before numbers so True/False never collapse into 1/0.
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, Mapping):
        parts = [f"{k!r}:{_structural_repr(value[k])}" for k in sorted(value, key=repr)]
        return "{" + ",".join(parts) + "}"
    # Bytes have no scalar normalizer -> stable repr key.
    if isinstance(value, bytes):
        return repr(value)
    if isinstance(value, Sequence | set | frozenset) and not isinstance(value, str):
        items = [_structural_repr(v) for v in value]
        # Preserve order for lists/tuples; sort sets for determinism.
        if isinstance(value, set | frozenset):
            items.sort()
        return "[" + ",".join(items) + "]"
    # Nested SCALAR leaf: reuse the top-level scalar normalizer so a nested
    # string/number/date gets the same case/whitespace, tolerance, and
    # instant normalization a top-level scalar would (e.g. {"name":"ACME"} and
    # {"name":"acme "} agree; "2020-01-02" and "Jan 2, 2020" agree). The
    # annotation is unknown here, so pass ``None`` and let value-based detection
    # (numeric-looking, date-like) drive it -- matching how an un-annotated
    # top-level scalar is canonicalized. Genuinely different values still differ.
    if isinstance(value, str | int | float | Decimal | _dt.date | _dt.datetime | _Enum):
        return _canon_scalar(value, None)
    return repr(value)


def normalize_value(leaf: LeafPath, value: Any) -> str:
    """Map a raw value at ``leaf`` to its ``normalized_key``.

    Returns :data:`MISSING_KEY` for anything in the unified absence bucket;
    otherwise a type-prefixed canonical string. Pure and deterministic.
    """
    if is_missing(value):
        return MISSING_KEY

    if leaf.kind is LeafKind.OBJECT_LIST_FIELD:
        # A keyed-row sub-value may itself be a dict / list (nested model or
        # container inside the row). Those must be canonicalized *structurally*
        # (sorted keys, recursive) so two semantically-equal payloads that differ
        # only in key order land in the same bucket; scalar repr would split them.
        if isinstance(value, Mapping) or (
            isinstance(value, Sequence | set | frozenset) and not isinstance(value, str | bytes)
        ):
            return _canon_structural(value)
        return _canon_scalar(value, leaf.annotation)
    if leaf.kind is LeafKind.SCALAR:
        return _canon_scalar(value, leaf.annotation)
    if leaf.kind is LeafKind.SCALAR_LIST:
        return _canon_scalar_list(value, leaf.annotation)
    # STRUCTURAL (dict / un-keyed list-of-objects).
    return _canon_structural(value)
