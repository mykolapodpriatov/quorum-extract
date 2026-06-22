"""Pydantic-v2 schema introspection: enumerate the *leaf paths* of a model.

A leaf path is the unit of voting. The grammar (plan 3.3) is explicit so that
every extractor normalizes and votes identically:

* **Scalar leaf** -- ``a.b`` for a plain scalar field. Unions of scalars are one
  leaf; the runtime value is compared directly.
* **Nested model** -- recurse into it, prefixing with ``parent.``. An *optional
  nested object that is absent at runtime* still contributes its leaves; they
  simply receive ``missing`` votes (handled at vote time, not here).
* **Array of scalars** (``list[int]``, ``set[str]``, ``tuple[float, ...]``) --
  ONE leaf, compared order-insensitively (not expanded).
* **Array of objects** (``list[SubModel]``) -- expands to ``field[*].sub``
  leaves **only when a ``list_key`` is declared** for ``field``. Without a
  declared key the whole list is a single structural leaf.
* **Dict / mapping** -- a single structural leaf (we do not invent per-key
  paths; that is out of scope).

The result of :func:`leaf_paths` is a stable, sorted list of :class:`LeafPath`
descriptors carrying the path string and enough type info for normalization.

This module uses only the Pydantic v2 public API (``model_fields``,
``FieldInfo.annotation``) plus :mod:`typing` introspection. It refuses Pydantic
v1 models with a clear error.
"""

from __future__ import annotations

import types as _types
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

# Container origins we treat as "array of X".
_SEQUENCE_ORIGINS: tuple[Any, ...] = (list, set, frozenset, tuple)
_MAPPING_ORIGINS: tuple[Any, ...] = (dict,)


class LeafKind(StrEnum):
    """What sort of leaf a path represents (drives normalization)."""

    SCALAR = "scalar"
    SCALAR_LIST = "scalar_list"
    """Order-insensitive list/set of scalars, one leaf."""
    STRUCTURAL = "structural"
    """A dict or an un-keyed list-of-objects: compared by structural equality."""
    OBJECT_LIST_FIELD = "object_list_field"
    """A ``field[*].sub`` leaf produced by expanding a keyed list-of-objects."""


@dataclass(frozen=True, slots=True)
class LeafPath:
    """A single votable leaf of a schema.

    Attributes:
        path: Dotted path string (e.g. ``"address.city"`` or
            ``"line_items[*].qty"``).
        kind: How to normalize/compare values at this path.
        annotation: The runtime type annotation of the scalar (for
            ``SCALAR`` / ``OBJECT_LIST_FIELD`` / ``SCALAR_LIST`` element type).
            ``None`` when not meaningful.
        list_field: For ``OBJECT_LIST_FIELD`` leaves, the owning list field path
            (e.g. ``"line_items"``); ``None`` otherwise.
        list_key: For ``OBJECT_LIST_FIELD`` leaves, the alignment key sub-field
            name; ``None`` otherwise.
    """

    path: str
    kind: LeafKind
    annotation: Any = None
    list_field: str | None = None
    list_key: str | None = None


class SchemaError(ValueError):
    """Raised when a schema cannot be introspected (e.g. Pydantic v1)."""


def _is_pydantic_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, was_optional)`` for ``Optional[X]`` / ``X | None``.

    Multi-member unions that are not simply ``X | None`` are returned unchanged
    (they are treated as scalar unions and compared by runtime value).
    """
    origin = get_origin(annotation)
    if origin is Union or origin is _types.UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
        # A genuine multi-type union (possibly including None): keep as-is.
        was_optional = len(non_none) != len(args)
        return annotation, was_optional
    return annotation, False


def _sequence_element(annotation: Any) -> Any | None:
    """If ``annotation`` is a supported sequence, return its element type.

    Returns ``None`` if it is not a sequence we expand.
    """
    origin = get_origin(annotation)
    if origin in _SEQUENCE_ORIGINS:
        args = get_args(annotation)
        if not args:
            return Any
        if origin is tuple:
            # Homogeneous tuple ``tuple[T, ...]`` -> element T; heterogeneous
            # tuples are treated structurally (return None to signal that).
            if len(args) == 2 and args[1] is Ellipsis:
                return args[0]
            return None
        return args[0]
    return None


def _ensure_v2(model: type[BaseModel]) -> None:
    if not hasattr(model, "model_fields"):  # pragma: no cover - defensive
        raise SchemaError(
            f"{model!r} does not look like a Pydantic v2 model "
            "(no `model_fields`). quorum-extract requires Pydantic >= 2.0."
        )


def leaf_paths(
    model: type[BaseModel],
    list_key: dict[str, str] | None = None,
    *,
    _prefix: str = "",
    _seen: frozenset[type[BaseModel]] | None = None,
) -> list[LeafPath]:
    """Enumerate the leaf paths of ``model`` per the documented grammar.

    Args:
        model: A Pydantic v2 ``BaseModel`` subclass.
        list_key: Maps a (fully-qualified) list field path to the sub-field name
            used to align rows; only such lists are expanded to ``field[*].sub``.
        _prefix / _seen: internal recursion state (do not pass).

    Returns:
        A list of :class:`LeafPath`, sorted by ``path`` for determinism.

    Raises:
        SchemaError: if ``model`` is not a Pydantic v2 model, or if a declared
            ``list_key`` references an unknown sub-field.
    """
    if not _is_pydantic_model(model):
        raise SchemaError(f"{model!r} is not a Pydantic BaseModel subclass")
    _ensure_v2(model)
    list_key = list_key or {}
    seen = _seen or frozenset()
    if model in seen:
        # Recursive/self-referential models: stop and emit one structural leaf
        # rather than recursing forever.
        return [LeafPath(path=_prefix.rstrip("."), kind=LeafKind.STRUCTURAL)]
    seen = seen | {model}

    resolved = _resolved_hints(model)
    out: list[LeafPath] = []
    for name, info in model.model_fields.items():
        path = f"{_prefix}{name}"
        # Prefer a fully-resolved hint (handles ForwardRef / string annotations
        # that arise under ``from __future__ import annotations`` or when a model
        # is loaded outside its defining module); fall back to the raw FieldInfo.
        annotation = resolved.get(name, info.annotation)
        out.extend(_leaf_for_annotation(path, annotation, list_key, seen))
    return sorted(out, key=lambda lp: lp.path)


def _resolved_hints(model: type[BaseModel]) -> dict[str, Any]:
    """Best-effort resolved type hints for ``model`` (empty dict on failure).

    Uses :func:`typing.get_type_hints` against the model's module namespace so
    forward references resolve. Falls back gracefully (returning what it can) so
    introspection never hard-fails on an exotic annotation.
    """
    import typing

    try:
        return typing.get_type_hints(model)
    except Exception:
        return {}


def _leaf_for_annotation(
    path: str,
    annotation: Any,
    list_key: dict[str, str],
    seen: frozenset[type[BaseModel]],
) -> list[LeafPath]:
    """Resolve one field annotation into one-or-more leaf paths."""
    inner, _optional = _strip_optional(annotation)

    # Nested Pydantic model -> recurse (optional-absent handled at vote time).
    if _is_pydantic_model(inner):
        return leaf_paths(inner, list_key, _prefix=f"{path}.", _seen=seen)

    # Mapping -> single structural leaf.
    if get_origin(inner) in _MAPPING_ORIGINS:
        return [LeafPath(path=path, kind=LeafKind.STRUCTURAL)]

    # Sequence -> scalar-list (one leaf) or object-list (maybe expand).
    element = _sequence_element(inner)
    if element is not None:
        elem_inner, _ = _strip_optional(element)
        if _is_pydantic_model(elem_inner):
            return _expand_object_list(path, elem_inner, list_key, seen)
        if (
            elem_inner is Any
            or _sequence_element(elem_inner) is not None
            or get_origin(elem_inner) in _MAPPING_ORIGINS
        ):
            # list of Any / nested containers -> structural.
            return [LeafPath(path=path, kind=LeafKind.STRUCTURAL)]
        return [LeafPath(path=path, kind=LeafKind.SCALAR_LIST, annotation=elem_inner)]
    # Heterogeneous tuple or other un-elementable sequence handled above as
    # element is None only for those -> structural.
    if get_origin(inner) is tuple:
        return [LeafPath(path=path, kind=LeafKind.STRUCTURAL)]

    # Plain scalar (or scalar union).
    return [LeafPath(path=path, kind=LeafKind.SCALAR, annotation=inner)]


def _expand_object_list(
    path: str,
    elem_model: type[BaseModel],
    list_key: dict[str, str],
    seen: frozenset[type[BaseModel]],
) -> list[LeafPath]:
    """Expand a list-of-objects, but only if a ``list_key`` is declared."""
    key = list_key.get(path)
    if key is None:
        # No declared key -> single structural leaf (compared as a whole).
        return [LeafPath(path=path, kind=LeafKind.STRUCTURAL)]

    _ensure_v2(elem_model)
    sub_fields = elem_model.model_fields
    if key not in sub_fields:
        raise SchemaError(
            f"list_key for {path!r} references sub-field {key!r} which is not a "
            f"field of {elem_model.__name__} (have: {sorted(sub_fields)})"
        )

    out: list[LeafPath] = []
    for sub_name, sub_info in sub_fields.items():
        sub_path = f"{path}[*].{sub_name}"
        sub_ann, _ = _strip_optional(sub_info.annotation)
        # We expand only one level: nested models inside list rows become
        # structural leaves (keeping alignment tractable and deterministic).
        if _is_pydantic_model(sub_ann) or get_origin(sub_ann) in (
            *_SEQUENCE_ORIGINS,
            *_MAPPING_ORIGINS,
        ):
            out.append(
                LeafPath(
                    path=sub_path,
                    kind=LeafKind.OBJECT_LIST_FIELD,
                    annotation=None,
                    list_field=path,
                    list_key=key,
                )
            )
        else:
            out.append(
                LeafPath(
                    path=sub_path,
                    kind=LeafKind.OBJECT_LIST_FIELD,
                    annotation=sub_ann,
                    list_field=path,
                    list_key=key,
                )
            )
    return out


def leaf_path_strings(model: type[BaseModel], list_key: dict[str, str] | None = None) -> list[str]:
    """Convenience: just the dotted path strings, sorted."""
    return [lp.path for lp in leaf_paths(model, list_key)]
