from dataclasses import replace
from fractions import Fraction as F
from graphlib import CycleError
from math import log

import pytest

from toricgt.symbolic_trajectory_balance import STOP, SymbolicDAG, Transition, subset_fixture


A = (("empty", "add-a", "a"),)
B = (("empty", "add-b", "b"),)
A_STOP = (*A, ("a", STOP, "done-a"))
AB_STOP = (*A, ("a", "add-b", "ab"), ("ab", STOP, "done-ab"))
BA_STOP = (*B, ("b", "add-a", "ab"), ("ab", STOP, "done-ab"))


def test_all_forward_masses_reverse_conditionals_and_terminal_rewards():
    dag = subset_fixture()
    expected = {A_STOP: (F(2, 7), F(1)), AB_STOP: (F(2, 7), F(2, 5)), BA_STOP: (F(3, 7), F(3, 5))}
    assert set(dag.paths()) == set(expected)
    for path, masses in expected.items():
        balance = dag.balance(path, partition=7)
        assert (balance.forward_mass, balance.backward_mass) == masses
        assert balance.ratio == 1 and balance.squared_log_residual == 0
    result = dag.audit(7)
    assert result["balanced_paths"] == result["paths"] == 3
    assert result["terminal_mass"] == result["normalized_rewards"] == {"done-a": F(2, 7), "done-ab": F(5, 7)}
    assert result["matches_rewards"]
    assert result["mean_squared_log_residual"] == 0
    wrong_partition = dag.audit(14)
    assert wrong_partition["balanced_paths"] == 0
    assert wrong_partition["mean_squared_log_residual"] == pytest.approx(log(2) ** 2)


@pytest.mark.parametrize("path", [
    (), A, AB_STOP[:-1], AB_STOP[1:], (*A_STOP, ("done-a", STOP, "done-a")),
    (*A, ("b", "add-a", "ab")), (("empty", "add-a", "b"),), (("empty", STOP, "done-a"),),
])
def test_rejects_incomplete_illegal_and_disconnected_paths(path):
    with pytest.raises(ValueError):
        subset_fixture().balance(path, 7)


# This graph also exercises multiple STOP predecessors of the same terminal.
EDGES = (
    Transition("s", "left", "l", F(1, 3), F(1)),
    Transition("s", "right", "r", F(2, 3), F(1)),
    Transition("l", STOP, "t", F(1), F(1, 3)),
    Transition("r", STOP, "t", F(1), F(2, 3)),
)


def test_normalized_reverse_kernel_can_still_disagree_with_forward_paths():
    assert SymbolicDAG("s", EDGES, {"t": 3}).audit(3)["balanced_paths"] == 2
    changed = (*EDGES[:2], replace(EDGES[2], backward=F(2, 3)), replace(EDGES[3], backward=F(1, 3)))
    result = SymbolicDAG("s", changed, {"t": 3}).audit(3)
    assert result["matches_rewards"] and result["balanced_paths"] == 0
    assert result["mean_squared_log_residual"] == pytest.approx(log(2) ** 2)


@pytest.mark.parametrize(("idx", "change", "match"), [
    (2, {"backward": F(1)}, "at successor t"),
    (0, {"forward": F(1)}, "at s"),
    (2, {"action": "finish"}, "STOP"),
    (0, {"action": STOP}, "STOP"),
    (0, {"forward": 0}, "positive"),
    (1, {"action": "left"}, "duplicate"),
])
def test_rejects_invalid_transition_tables(idx, change, match):
    edges = list(EDGES)
    edges[idx] = replace(edges[idx], **change)
    with pytest.raises(ValueError, match=match):
        SymbolicDAG("s", tuple(edges), {"t": 1})


def test_rejects_unreachable_cycle_or_missing_positive_reward():
    with pytest.raises(CycleError):
        SymbolicDAG("s", (*EDGES, Transition("x", "loop", "x", F(1), F(1))), {"t": 1})
    with pytest.raises(ValueError, match="positive"):
        SymbolicDAG("s", EDGES, {"t": 0})
    with pytest.raises(ValueError, match="at successor unreachable"):
        SymbolicDAG("s", EDGES, {"t": 2, "unreachable": 1})
    with pytest.raises(ValueError, match="positive"):
        subset_fixture().balance(A_STOP, 0)
