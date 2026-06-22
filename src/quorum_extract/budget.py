"""Cost accounting and the ``$ saved`` value proposition.

Costs are per *invocation* (one call extracts the whole record). The budget
tracker accumulates:

* ``cheap_cost`` -- sum of cheap-tier invocation costs (every doc x every cheap
  extractor).
* ``escalation_cost`` -- one strong invocation per *doc-with-contention*.
* ``all_frontier_cost`` -- the hypothetical of running the strong extractor on
  *every* document.

``saved = all_frontier_cost - escalation_cost`` is the measurable value
proposition: how much was *not* spent by escalating only contested docs instead
of frontier-everything.

A :class:`BudgetTracker` also enforces an optional ``max_cost_usd`` cap on
escalation spend; documents are escalated in a deterministic order (by doc id)
so the cap point is reproducible, and anything past it is marked for review
upstream (never dropped).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import BudgetReport


@dataclass
class BudgetTracker:
    """Accumulates run costs and decides whether escalation may proceed.

    Args:
        strong_cost_usd: Cost of one strong (escalation) invocation. Used both to
            charge escalations and to compute the all-frontier hypothetical.
        max_cost_usd: Optional cap on *escalation* spend. ``None`` = unlimited.
    """

    strong_cost_usd: float
    max_cost_usd: float | None = None
    cheap_cost_usd: float = 0.0
    escalation_cost_usd: float = 0.0
    docs_total: int = 0
    docs_escalated: int = 0
    docs_over_budget: int = 0
    _doc_ids: list[str] = field(default_factory=list)

    def charge_cheap(self, doc_id: str, cost_usd: float) -> None:
        """Record the cheap-extraction cost for a document."""
        if doc_id not in self._doc_ids:
            self._doc_ids.append(doc_id)
            self.docs_total += 1
        self.cheap_cost_usd += cost_usd

    def can_escalate(self) -> bool:
        """True iff charging one more strong invocation stays within the cap."""
        if self.max_cost_usd is None:
            return True
        return self.escalation_cost_usd + self.strong_cost_usd <= self.max_cost_usd + 1e-12

    def charge_escalation(self) -> bool:
        """Attempt to charge one strong invocation.

        Returns ``True`` and records the cost if within budget; otherwise returns
        ``False``, increments the over-budget counter, and charges nothing.
        """
        if not self.can_escalate():
            self.docs_over_budget += 1
            return False
        self.escalation_cost_usd += self.strong_cost_usd
        self.docs_escalated += 1
        return True

    @property
    def total_cost_usd(self) -> float:
        return self.cheap_cost_usd + self.escalation_cost_usd

    @property
    def all_frontier_cost_usd(self) -> float:
        """Hypothetical cost of running the strong extractor on every document."""
        return self.strong_cost_usd * self.docs_total

    @property
    def saved_usd(self) -> float:
        """``all_frontier_cost - escalation_cost`` (never negative in practice).

        This is the headline: we paid for strong invocations only on contested
        docs instead of on all of them.
        """
        return self.all_frontier_cost_usd - self.escalation_cost_usd

    def report(self) -> BudgetReport:
        """Snapshot the accumulated costs as a :class:`BudgetReport`."""
        return BudgetReport(
            cheap_cost_usd=round(self.cheap_cost_usd, 10),
            escalation_cost_usd=round(self.escalation_cost_usd, 10),
            total_cost_usd=round(self.total_cost_usd, 10),
            all_frontier_cost_usd=round(self.all_frontier_cost_usd, 10),
            saved_usd=round(self.saved_usd, 10),
            docs_total=self.docs_total,
            docs_escalated=self.docs_escalated,
            docs_over_budget=self.docs_over_budget,
        )
