"""Per-field quorum: bucket normalized votes and pick the winner.

Given, for one document, each extractor's raw output dict, this module:

1. Resolves a value for every leaf path (including expanding keyed
   list-of-objects into ``field[*].sub`` votes aligned by the declared key).
2. Normalizes each value into a bucket (:mod:`quorum_extract.normalize`), with
   a unified ``missing`` bucket and ``K`` preserved across failed extractors.
3. Computes ``agreement = winning_bucket_size / K`` with a deterministic
   tie-break (lowest extractor tier, then earliest extractor order).

The list-alignment rule (plan 3.3) is strict and deterministic: a model with a
**duplicate key**, a **missing key on a row**, or a **key set that doesn't
match** the other models contributes a **full disagreement** for that list's
affected leaves -- "full disagreement" is *asserted* by the algorithm, never
guessed by position.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .normalize import ABSENT, MISSING_KEY, is_missing, normalize_value
from .schema import LeafKind, LeafPath
from .types import (
    EscalationStatus,
    ExtractorSpec,
    FieldResult,
    FieldVote,
    QuorumConfig,
)

# Sentinel bucket key marking an alignment-induced full disagreement. Each such
# vote gets a *unique* key so the votes never coalesce into agreement.
_DISAGREE_PREFIX = "__disagree__:"


class _ListAbsent:
    """Sentinel type for "the object-list field was absent or None".

    Distinct from a *malformed* list (a ``None`` index from a duplicate/missing
    key): an absent/None list is a unanimous-able ``missing`` vote -- two
    extractors that both omit the list AGREE it is absent (agreement 1.0) -- and
    must NOT be routed to the unique full-disagreement path.
    """

    __slots__ = ()


#: Singleton instance of :class:`_ListAbsent`.
LIST_ABSENT = _ListAbsent()

# Per-list, per-extractor row index: a clean ``dict`` index, ``None`` for a
# malformed list (duplicate/missing key), or :data:`LIST_ABSENT` when the list
# field itself was absent/None.
RowIndex = dict[Any, Any] | None | _ListAbsent
ListIndexMap = dict[str, dict[str, RowIndex]]


@dataclass(frozen=True, slots=True)
class ExtractorOutput:
    """One extractor's raw output for one document.

    ``ok=False`` marks a failed/timed-out extractor; its data is ignored and
    every field receives a ``missing`` vote (K is still preserved).
    """

    spec: ExtractorSpec
    data: Mapping[str, Any]
    ok: bool = True


def _dig(data: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a dotted path inside a nested mapping, returning ABSENT if any
    segment is missing. Only plain-mapping traversal (lists handled separately).
    """
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return ABSENT
    return cur


def _resolve_scalar(output: ExtractorOutput, leaf: LeafPath) -> Any:
    """Resolve a scalar / scalar-list / structural leaf's raw value."""
    if not output.ok:
        return ABSENT
    return _dig(output.data, leaf.path)


def _votes_for_simple_leaf(leaf: LeafPath, outputs: Sequence[ExtractorOutput]) -> list[FieldVote]:
    votes: list[FieldVote] = []
    for out in outputs:
        raw = _resolve_scalar(out, leaf)
        key = normalize_value(leaf, raw)
        votes.append(
            FieldVote(
                extractor=out.spec.name,
                raw_value=None if raw is ABSENT else raw,
                normalized_key=key,
                missing=key == MISSING_KEY,
            )
        )
    return votes


def _extract_rows(value: Any) -> list[Any] | None:
    """Coerce a list-of-objects value into a list of row mappings, or None."""
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return None


def _row_index_by_key(rows: list[Any], key: str) -> dict[Any, Any] | None:
    """Index rows by their key value; return None if any row lacks a unique,
    present key (signals full disagreement for this extractor's list).
    """
    index: dict[Any, Any] = {}
    seen: set[Any] = set()
    for row in rows:
        if not isinstance(row, Mapping) or key not in row:
            return None
        kv = row[key]
        if kv is None or (isinstance(kv, str) and kv.strip() == ""):
            return None
        # Make the key hashable & comparable; fall back to repr for unhashables.
        try:
            hkey = kv if isinstance(kv, str | int | float | bool) else repr(kv)
        except Exception:  # pragma: no cover - defensive
            return None
        if hkey in seen:
            return None  # duplicate key -> full disagreement
        seen.add(hkey)
        index[hkey] = row
    return index


def _votes_for_object_list_field(
    leaf: LeafPath,
    outputs: Sequence[ExtractorOutput],
    list_field_indices: ListIndexMap,
) -> list[FieldVote]:
    """Build votes for one ``field[*].sub`` leaf using precomputed row indices.

    Alignment rule: a sub-leaf agrees across extractors only on rows whose key
    is present and unique in *all* compared extractors. An extractor whose list
    is *malformed* (dup/missing key) votes a unique full-disagreement key; an
    extractor whose list field is *absent/None* votes ``missing`` (so two
    extractors that both omit the list agree it is absent).
    """
    assert leaf.list_field is not None and leaf.list_key is not None
    sub = leaf.path.split("[*].", 1)[1]
    votes: list[FieldVote] = []

    indices = list_field_indices[leaf.list_field]
    # The set of keys present-and-unique across every extractor that has a clean
    # index. If indices disagree on which keys exist, only the intersection is
    # comparable; rows outside it count as disagreement.
    common_keys: set[Any] | None = None
    for idx in indices.values():
        if isinstance(idx, dict):
            ks = set(idx.keys())
            common_keys = ks if common_keys is None else (common_keys & ks)
    if common_keys is None:
        common_keys = set()

    for out in outputs:
        name = out.spec.name
        idx = indices.get(name)
        if not out.ok or isinstance(idx, _ListAbsent):
            # Failed extractor OR an absent/None list field -> unified ``missing``
            # vote (the same bucket a failed extractor lands in), so unanimous
            # absence yields agreement 1.0 rather than distinct disagreement keys.
            votes.append(
                FieldVote(extractor=name, raw_value=None, normalized_key=MISSING_KEY, missing=True)
            )
            continue
        if idx is None:
            # Malformed list (dup/missing key) -> unique full disagreement.
            votes.append(
                FieldVote(
                    extractor=name,
                    raw_value=None,
                    normalized_key=f"{_DISAGREE_PREFIX}{name}",
                    missing=False,
                )
            )
            continue
        # A clean extractor whose key set differs from the common set still
        # disagrees on the non-common rows; for the leaf as a whole we compare on
        # the union of keys. We aggregate each extractor's per-row sub values
        # into a single deterministic key over the COMMON keys, and append a
        # disagreement marker if its key set isn't exactly the common set.
        if set(idx.keys()) != common_keys:
            votes.append(
                FieldVote(
                    extractor=name,
                    raw_value=None,
                    normalized_key=f"{_DISAGREE_PREFIX}{name}",
                    missing=False,
                )
            )
            continue
        parts: list[str] = []
        norms: list[str] = []
        for k in sorted(common_keys, key=repr):
            row = idx[k]
            raw = row.get(sub, ABSENT) if isinstance(row, Mapping) else ABSENT
            norm = normalize_value(leaf, raw)
            norms.append(norm)
            parts.append(f"{k!r}={norm}")
        agg_key = "rows:{" + ",".join(parts) + "}"
        # No comparable rows (a clean but EMPTY keyed list -- ``[]`` -- yields an
        # empty index and hence ``not norms``) means "no rows", semantically the
        # same as an absent/None list. Unify it into the ``missing`` bucket so an
        # extractor returning ``[]`` agrees with one returning an absent/None list
        # (the empty list is a clean index, NOT the malformed ``idx is None`` path
        # above, which stays a unique ``__disagree__`` disagreement).
        #
        # Otherwise: exact equality on each row's bare normalized key -- NOT a
        # suffix check on the composite ``'k'=norm`` string -- so a real value
        # whose canonical form merely ends with ``__missing__`` (e.g.
        # ``"hello__missing__"``) is not mis-classified as absent.
        is_missing = not norms or all(n == MISSING_KEY for n in norms)
        votes.append(
            FieldVote(
                extractor=name,
                raw_value=None,
                normalized_key=MISSING_KEY if is_missing else agg_key,
                missing=is_missing,
            )
        )
    return votes


def _build_list_indices(
    outputs: Sequence[ExtractorOutput], leaves: Sequence[LeafPath]
) -> ListIndexMap:
    """Precompute, per list field, each extractor's row index.

    Per extractor the value is a clean ``dict`` index, ``None`` for a *malformed*
    list (duplicate/missing key, or a present-but-non-sequence value), or
    :data:`LIST_ABSENT` when the list field is absent/``None`` (an absence that
    unifies into the ``missing`` bucket).
    """
    list_fields: dict[str, str] = {}
    for leaf in leaves:
        if leaf.kind is LeafKind.OBJECT_LIST_FIELD and leaf.list_field and leaf.list_key:
            list_fields[leaf.list_field] = leaf.list_key

    result: ListIndexMap = {}
    for field_path, key in list_fields.items():
        per_extractor: dict[str, RowIndex] = {}
        for out in outputs:
            if not out.ok:
                per_extractor[out.spec.name] = LIST_ABSENT
                continue
            raw = _dig(out.data, field_path)
            if is_missing(raw):
                # Absent key / None / empty-string list value -> missing, not a
                # malformed-list disagreement.
                per_extractor[out.spec.name] = LIST_ABSENT
                continue
            rows = _extract_rows(raw)
            if rows is None:
                per_extractor[out.spec.name] = None
            else:
                per_extractor[out.spec.name] = _row_index_by_key(rows, key)
        result[field_path] = per_extractor
    return result


def collect_votes(
    leaf: LeafPath,
    outputs: Sequence[ExtractorOutput],
    list_indices: ListIndexMap | None = None,
) -> list[FieldVote]:
    """Public helper: votes for a single leaf across all extractor outputs."""
    if leaf.kind is LeafKind.OBJECT_LIST_FIELD:
        if list_indices is None:
            list_indices = _build_list_indices(outputs, [leaf])
        return _votes_for_object_list_field(leaf, outputs, list_indices)
    return _votes_for_simple_leaf(leaf, outputs)


def _tier_of(name: str, outputs: Sequence[ExtractorOutput]) -> tuple[int, int]:
    """Return ``(tier, order)`` for an extractor name for tie-breaking."""
    for i, out in enumerate(outputs):
        if out.spec.name == name:
            return out.spec.tier, i
    return 0, len(outputs)  # pragma: no cover - name always present


def tally(votes: Sequence[FieldVote], outputs: Sequence[ExtractorOutput]) -> tuple[str, int, int]:
    """Tally votes into ``(winning_key, winning_count, K)``.

    Deterministic tie-break among equal-count buckets: prefer the bucket whose
    *best* (lowest-tier, then earliest) contributing extractor ranks first.
    ``missing`` is an ordinary bucket here and can win.
    """
    counts: Counter[str] = Counter(v.normalized_key for v in votes)
    k = len(votes)
    # Best (tier, order) seen per bucket -> tie-break preference.
    best_rank: dict[str, tuple[int, int]] = {}
    for v in votes:
        rank = _tier_of(v.extractor, outputs)
        cur = best_rank.get(v.normalized_key)
        if cur is None or rank < cur:
            best_rank[v.normalized_key] = rank

    def sort_key(item: tuple[str, int]) -> tuple[int, tuple[int, int]]:
        key, count = item
        return (-count, best_rank[key])

    winning_key, winning_count = min(counts.items(), key=sort_key)
    return winning_key, winning_count, k


def _value_for_key(key: str, votes: Sequence[FieldVote], outputs: Sequence[ExtractorOutput]) -> Any:
    """Pick the representative raw value for the winning bucket.

    Among votes in the bucket, choose the one from the lowest-tier / earliest
    extractor (deterministic). ``missing`` resolves to ``None``.
    """
    if key == MISSING_KEY:
        return None
    candidates = [v for v in votes if v.normalized_key == key]
    candidates.sort(key=lambda v: _tier_of(v.extractor, outputs))
    return candidates[0].raw_value if candidates else None


def quorum_field(
    leaf: LeafPath,
    outputs: Sequence[ExtractorOutput],
    config: QuorumConfig,
    list_indices: ListIndexMap | None = None,
) -> FieldResult:
    """Reconcile one leaf path into a :class:`FieldResult` (pre-calibration).

    The returned status is ``accepted`` iff ``agreement >= min_agreement`` (and
    the winning bucket is not ``missing`` -- a field that everyone agrees is
    absent is accepted as ``None`` but still meets quorum). Calibration gating
    (``min_confidence``) and escalation are applied later by the cascade.
    """
    votes = collect_votes(leaf, outputs, list_indices)
    winning_key, winning_count, k = tally(votes, outputs)
    agreement = winning_count / k if k else 0.0
    value = _value_for_key(winning_key, votes, outputs)
    status = (
        EscalationStatus.ACCEPTED
        if agreement >= config.min_agreement
        else EscalationStatus.NEEDS_REVIEW
    )
    return FieldResult(
        path=leaf.path,
        value=value,
        votes=votes,
        agreement=agreement,
        confidence=None,
        status=status,
        winning_key=winning_key,
    )


def quorum_record(
    leaves: Sequence[LeafPath],
    outputs: Sequence[ExtractorOutput],
    config: QuorumConfig,
) -> dict[str, FieldResult]:
    """Reconcile every leaf path for one document.

    Guarantees the field-completeness invariant at the quorum stage: the result
    has exactly one :class:`FieldResult` per leaf path.
    """
    list_indices = _build_list_indices(outputs, leaves)
    results: dict[str, FieldResult] = {}
    for leaf in leaves:
        results[leaf.path] = quorum_field(leaf, outputs, config, list_indices)
    return results
