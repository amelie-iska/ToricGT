from dataclasses import asdict, replace
from itertools import product

import pytest

from toricgt.toricmora_symbolic_search import (
    Caps, METHODS, PROBLEMS, Problem, analogy_preferences, benchmark, propose, search, verifies,
)


# Exhaustive, hand-enumerated solution sets derived from the infix questions,
# independent of search traversal, structural ranking and the stack verifier.
GOLD = {
    "ordered-left": {("*", "+")},
    "ordered-right": {("*", "-")},
    "paired-roots": {("*", "*", "-")},
    "unseen-balanced-shape": {("*", "-", "*")},
    "ambiguous": {("+",), ("*",)},
    "infeasible": set(),
}


def test_verifier_matches_independent_solution_sets():
    for problem in PROBLEMS:
        assignments = product(("+", "-", "*"), repeat=problem.hole_count)
        assert {prefix for prefix in assignments if verifies(problem, prefix)} == GOLD[problem.name]
        assert not verifies(problem, ())
        assert not verifies(problem, ("/",) * problem.hole_count)
    original = PROBLEMS[0]
    assert not verifies(original, ("*", "+", "-"))
    assert not verifies(replace(original, targets=(16,)), ("*", "+"))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("caps", [Caps(), Caps(2, 4, 0), Caps(0, 0, 0), Caps(40, 1, 3)])
def test_replay_transitions_accounting_and_selection(method, caps):
    for problem in PROBLEMS:
        report = search(problem, method, caps=caps)
        reached = {i: (i, ()) for i in range(report["roots"])}
        expanded, verified, rejected, unchecked = set(), {}, set(), set()
        used = report["used"]
        for event in report["events"]:
            parent = event["parent"]
            parent_id, tree_id, prefix = parent["id"], parent["tree_id"], parent["prefix"]
            assert parent_id not in expanded
            assert reached[parent_id] == (tree_id, prefix)
            assert event["parent_depth"] == len(prefix) < problem.hole_count
            expanded.add(parent_id)
            for transition in event["transitions"]:
                action = transition["proposed_action"]
                child = transition["executed_child"]
                assert action in {"+", "-", "*"}
                assert child["tree_id"] == tree_id and child["prefix"] == (*prefix, action)
                assert child["id"] not in reached
                reached[child["id"]] = (tree_id, child["prefix"])
                status = transition["status"]
                if len(child["prefix"]) == problem.hole_count:
                    if status == "unverified_budget":
                        unchecked.add(child["id"])
                    elif child["prefix"] in GOLD[problem.name]:
                        assert status == "verified"
                        verified[child["id"]] = child["prefix"]
                    else:
                        assert status == "rejected"
                        rejected.add(child["id"])
                else:
                    assert status == "open"
        assert used == {
            "tokens": len(reached) - report["roots"], "expansions": len(expanded),
            "verifier_calls": len(verified) + len(rejected),
        }
        assert all(used[key] <= cap for key, cap in asdict(caps).items())
        leaves = {leaf["node"]["id"]: leaf["status"] for leaf in report["leaves"]}
        assert len(leaves) == len(report["leaves"])
        assert leaves.keys() == reached.keys() - expanded
        assert {i for i, s in leaves.items() if s == "unverified_budget"} == unchecked
        assert set(report["solutions"]) == set(verified.values())
        assert report["selected_node"] == next(iter(verified), None)
        assert report["selected"] == next(iter(verified.values()), None)
        assert report["observed_ambiguity"] == (len(set(verified.values())) > 1)


def test_branching_scheduling_and_analogy():
    problem = PROBLEMS[2]
    calls = {method: search(problem, method, caps=Caps(1000, 1000, 1000)) for method in METHODS}
    for method in ("single_path", "best_of_n"):
        assert all(len(e["transitions"]) == 1 for e in calls[method]["events"])
    assert calls["single_path"]["roots"] == calls["tot"]["roots"] == 1
    assert len(calls["best_of_n"]["leaves"]) == 3
    for method in ("tot", "fot", "fot_analogy"):
        assert all(len(e["transitions"]) == 3 for e in calls[method]["events"])
        assert set(calls[method]["solutions"]) == GOLD[problem.name]
    for method in ("fot", "fot_analogy"):
        assert [e["parent"]["tree_id"] for e in calls[method]["events"][:6]] == [0, 1, 2] * 2
    preferences = analogy_preferences(problem)
    assert [votes.most_common(1)[0][0] for votes in preferences] == ["*", "*", "-"]
    assert all(e["transitions"][0]["proposed_action"] == preferences[e["parent_depth"]].most_common(1)[0][0]
               for e in calls["fot_analogy"]["events"])


@pytest.mark.parametrize("method", METHODS)
def test_rejected_terminals_cannot_change_preceding_or_sibling_proposals(method):
    problem = PROBLEMS[2]
    changed = replace(problem, targets=(0, 999))

    def emitted(report):
        return [(e["parent"], [t["executed_child"] for t in e["transitions"]]) for e in report["events"]]

    assert emitted(search(changed, method)) == emitted(search(problem, method))


def test_exhaustion_ambiguity_and_infeasibility():
    ambiguous, infeasible = PROBLEMS[-2:]
    full = search(ambiguous, "tot", caps=Caps(3, 1, 3))
    assert full["enumeration_complete"] and full["verdict"] == "multiple_verified"
    assert full["observed_ambiguity"] and full["budget_exhausted"] == []
    incomplete = search(ambiguous, "tot", caps=Caps(3, 1, 0))
    assert incomplete["selected"] is None and incomplete["budget_exhausted"] == ["verifier_calls"]
    exhaustive = search(infeasible, "tot")
    assert exhaustive["verdict"] == "infeasible_proven"
    assert exhaustive["solutions"] == [] and exhaustive["budget_exhausted"] == []
    clipped = search(ambiguous, "tot", caps=Caps(1, 1, 1))
    assert clipped["budget_exhausted"] == ["tokens"]
    assert clipped["events"][0]["unemitted_actions"] == 2


def test_all_five_methods_use_the_same_prompt_caps_and_seed():
    report = benchmark(seed=29, caps=Caps(11, 5, 2))
    assert len(report["runs"]) == len(METHODS) * len(PROBLEMS)
    for run in report["runs"]:
        assert run["caps"] == report["caps_per_case"]
        assert run["seed"] == 29
        assert "events" not in run and "leaves" not in run
    assert not analogy_preferences(PROBLEMS[3])[-1]
    for problem in PROBLEMS:
        first = tuple(t["proposed_action"] for t in search(problem, "tot")["events"][0]["transitions"])
        assert list(first) == propose(17, 0, ())


@pytest.mark.parametrize("tree", [(2, 2), (2, "?"), (2, 2, "%")])
def test_invalid_postfix_tree(tree):
    with pytest.raises(ValueError):
        Problem("invalid", (tree,), (4,))
