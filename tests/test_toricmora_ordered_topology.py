from itertools import product
import json
import operator
import sys

import pytest

from toricgt import toricmora_symbolic_search as search


LEAF, CLOSE = search.ADD_OPERAND, search.CLOSE
ADD, SUB, MUL = ("ADD_CHILD " + op for op in "+-*")
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
LARGE = search.Caps(tokens=5000, expansions=5000, verifier_calls=500)


def replay(operands, actions):
    state = search.OrderedPrefix()
    for action in actions:
        state = state.advance(action, operands)
    return state


def test_ordered_shapes_and_terminal_arithmetic_against_independent_oracle():
    problem = search.Problem("shape", ((2, 3, "?", 4, "?"),), (14,))
    result = search.search_ordered_topology(problem, caps=LARGE, traces=True)
    # All 2 Catalan shapes x 3^2 operator choices are visited exactly once.
    assert result["used"]["verifier_calls"] == 18
    assert result["budget_exhausted"] == [] and result["complete_path_length"] == 10
    gold = set()
    for root, child in product(OPS, repeat=2):
        if OPS[root](OPS[child](2, 3), 4) == 14:
            gold.add(((root, (child, 2, 3), 4),))
        if OPS[root](2, OPS[child](3, 4)) == 14:
            gold.add(((root, 2, (child, 3, 4)),))
    assert gold == {(("*", 2, ("+", 3, 4)),), (("+", 2, ("*", 3, 4)),)}
    assert set(result["solutions"]) == gold
    for event in result["events"]:
        state = replay(result["operands"], event["prefix"])
        assert event["ancestors"] == state.ancestors
        assert event["action"] in state.legal(result["operands"])
    assert result["used"]["tokens"] == len(result["events"])


def test_forest_closes_roots_in_order_and_rejects_junk_or_partial_terminals():
    operands = ((7, 2), (3,))
    actions = (SUB, LEAF, CLOSE, LEAF, CLOSE, CLOSE, LEAF, CLOSE)
    assert search.verify_ordered_terminal(actions, operands, (5, 3)) == (("-", 7, 2), 3)
    assert search.verify_ordered_terminal(actions, operands, (-5, 3)) is None
    assert search.verify_ordered_terminal(actions, operands, (3, 5)) is None
    for invalid in (actions[1:], actions[:-1], (*actions, CLOSE), (*actions, None)):
        assert search.verify_ordered_terminal(invalid, operands, (5, 3)) is None
    state = replay(operands, actions)
    assert state.root == 2 and state.ancestors == () and state.legal(operands) == ()
    with pytest.raises(ValueError, match="illegal"):
        state.advance(ADD, operands)


def test_ranker_receives_only_open_ancestor_labels():
    left = ((2, 3, 4, 5),)
    right = ((70, 80, 4, 99),)
    prefix = (ADD, MUL, LEAF, CLOSE, LEAF, CLOSE, CLOSE, SUB)
    changed = (ADD, SUB, *prefix[2:])
    contexts = []
    for operands, path in ((left, prefix), (right, changed)):
        state = replay(operands, path)
        assert state.ancestors == ("+", "-")
        contexts.append((search.rank_topology(state.ancestors, seed=91), state.legal(operands)))
    assert contexts[0] == contexts[1]

    seen = []
    def spy(ancestors, *, seed):
        assert type(ancestors) is tuple and seed == 91
        seen.append(ancestors)
        return search.TOPOLOGY_ACTIONS

    shared = dict(ranker=spy, caps=search.Caps(20, 10, 3), seed=91, traces=True)
    original = search.Problem("one", ((2, 3, "?", 4, "?"),), (14,))
    result = search.search_ordered_topology(original, **shared)
    calls = list(seen)
    seen.clear()
    # Same source leaves, different hidden grouping and terminal answer.
    changed = search.Problem("two", ((2, 3, 4, "?", "?"),), (99,))
    other = search.search_ordered_topology(changed, **shared)
    assert seen == calls
    assert [e["action"] for e in result["events"]] == [e["action"] for e in other["events"]]


@pytest.mark.parametrize("caps", [search.Caps(0, 3, 2), search.Caps(4, 0, 3),
                                  search.Caps(3, 10, 4), search.Caps(40, 40, 0)])
def test_each_cap_is_enforced(caps):
    result = search.search_ordered_topology(search.PROBLEMS[-1], caps=caps, traces=True)
    assert result["budget_exhausted"]
    for key, count in result["used"].items():
        assert count <= getattr(caps, key)
    if caps.verifier_calls == 0:
        assert result["selected"] is None
        assert any(e["status"] == "unverified_budget" for e in result["events"])


def test_optional_cli_mode_and_legacy_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["symbolic-search", "--ordered-topology", "--traces"])
    search.main()
    report = json.loads(capsys.readouterr().out)
    assert report["fixture"] == "ordered-topology-v1"
    assert len(report["runs"]) == len(search.PROBLEMS)
    assert all("events" in run for run in report["runs"])
    monkeypatch.setattr(sys, "argv", ["symbolic-search"])
    search.main()
    assert json.loads(capsys.readouterr().out)["fixture"] == search.FIXTURE_VERSION
