"""Tests for cost accounting and the $ saved value proposition."""

from __future__ import annotations

from quorum_extract import BudgetTracker


def test_charge_cheap_counts_docs_once() -> None:
    t = BudgetTracker(strong_cost_usd=0.05)
    t.charge_cheap("d1", 0.003)
    t.charge_cheap("d1", 0.003)  # same doc again (e.g. re-quorum) -> still 1 doc
    t.charge_cheap("d2", 0.003)
    assert t.docs_total == 2
    assert abs(t.cheap_cost_usd - 0.009) < 1e-12


def test_escalation_charges_strong_cost() -> None:
    t = BudgetTracker(strong_cost_usd=0.05)
    t.charge_cheap("d1", 0.003)
    assert t.charge_escalation() is True
    assert t.escalation_cost_usd == 0.05
    assert t.docs_escalated == 1


def test_budget_cap_blocks_escalation() -> None:
    t = BudgetTracker(strong_cost_usd=0.05, max_cost_usd=0.05)
    for d in ("d1", "d2", "d3"):
        t.charge_cheap(d, 0.001)
    assert t.charge_escalation() is True  # first within budget
    assert t.charge_escalation() is False  # second exceeds cap
    assert t.docs_escalated == 1
    assert t.docs_over_budget == 1
    assert t.escalation_cost_usd == 0.05  # nothing charged for the blocked one


def test_unlimited_budget_when_cap_none() -> None:
    t = BudgetTracker(strong_cost_usd=0.05, max_cost_usd=None)
    for _ in range(10):
        assert t.charge_escalation() is True
    assert t.docs_escalated == 10


def test_saved_usd_is_all_frontier_minus_escalation() -> None:
    t = BudgetTracker(strong_cost_usd=0.05)
    for d in ("d1", "d2", "d3", "d4"):
        t.charge_cheap(d, 0.001)
    t.charge_escalation()  # only one doc contested
    rep = t.report()
    assert rep.all_frontier_cost_usd == 0.2  # 0.05 * 4
    assert rep.escalation_cost_usd == 0.05
    assert rep.saved_usd == 0.15
    assert abs(rep.total_cost_usd - 0.054) < 1e-9


def test_report_snapshot_fields() -> None:
    t = BudgetTracker(strong_cost_usd=0.1)
    t.charge_cheap("d1", 0.01)
    t.charge_escalation()
    rep = t.report()
    assert rep.docs_total == 1
    assert rep.docs_escalated == 1
    assert rep.docs_over_budget == 0


def test_can_escalate_predicate() -> None:
    t = BudgetTracker(strong_cost_usd=0.05, max_cost_usd=0.05)
    assert t.can_escalate() is True
    t.charge_escalation()
    assert t.can_escalate() is False
