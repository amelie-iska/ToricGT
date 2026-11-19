"""Bounded prefix search on authored ordered arithmetic forests.

Each source tree is postfix notation. A question mark is a binary operator
hole. Each action generates one operator token, in forest and postfix order.
Public targets are constraints of the puzzle. The proposer receives no gold
operator assignments, verifier decisions, or sibling continuations.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import operator
import random
from time import perf_counter

OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
METHODS = ("single_path", "best_of_n", "tot", "fot", "fot_analogy")
FIXTURE_VERSION = "ordered-operator-forest-v1"
# Same worked example in every method's prompt. Only fot_analogy uses it
# when ordering proposals. Integers in the worked example are disjoint from
# every query. The target assignment is never an input to the proposer.
EXEMPLAR = ((31, 37, "*", 41, "+"), (43, 47, 53, "*", "-"))
Tree = tuple[int | str, ...]
Prefix = tuple[str, ...]


def _shapes(tree: Tree, allowed: tuple[str, ...]) -> tuple[tuple, ...]:
    """Return the ordered, value-independent subtree shape of each operator."""
    stack = []
    shapes = []
    for token in tree:
        if type(token) is int:
            stack.append(())
        elif token in allowed and len(stack) >= 2:
            right = stack.pop()
            left = stack.pop()
            shape = (left, right)
            shapes.append(shape)
            stack.append(shape)
        else:
            raise ValueError("expected a postfix binary tree")
    if len(stack) != 1:
        raise ValueError("each tree must have exactly one root")
    return tuple(shapes)


@dataclass(frozen=True)
class Problem:
    name: str
    trees: tuple[Tree, ...]
    targets: tuple[int, ...]

    def __post_init__(self):
        if not self.trees or len(self.targets) != len(self.trees):
            raise ValueError("one target is required per ordered tree")
        if any(type(value) is not int for value in self.targets):
            raise ValueError("targets must be integers")
        if not any("?" in tree for tree in self.trees):
            raise ValueError("a problem needs at least one operator hole")
        for tree in self.trees:
            _shapes(tree, ("?",))

    @property
    def hole_count(self):
        return sum(tree.count("?") for tree in self.trees)


PROBLEMS = (
    Problem("ordered-left", ((3, 4, "?", 5, "?"),), (17,)),
    Problem("ordered-right", ((12, 2, 3, "?", "?"),), (6,)),
    Problem("paired-roots", ((3, 5, "?"), (9, 2, 4, "?", "?")), (15, 1)),
    Problem("unseen-balanced-shape", ((2, 5, "?", 8, 3, "?", "?"),), (50,)),
    Problem("ambiguous", ((2, 2, "?"),), (4,)),
    Problem("infeasible", ((3, 4, "?"),), (8,)),
)


def verifies(problem: Problem, prefix: Prefix) -> bool:
    """Evaluate complete operator assignments with an ordered stack machine."""
    if len(prefix) != problem.hole_count or any(op not in OPERATORS for op in prefix):
        return False
    actions = iter(prefix)
    for tree, target in zip(problem.trees, problem.targets):
        values = []
        for token in tree:
            if token == "?":
                right = values.pop()
                left = values.pop()
                values.append(OPERATORS[next(actions)](left, right))
            else:
                values.append(token)
        if values != [target]:
            return False
    return True


@dataclass(frozen=True)
class Caps:
    tokens: int = 24
    expansions: int = 12
    verifier_calls: int = 8

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in asdict(self).values()):
            raise ValueError("caps must be nonnegative integers")


def propose(seed: int, tree_id: int, prefix: Prefix) -> list[str]:
    """Seeded prefix policy shared by all methods, with stable RNG per root."""
    material = json.dumps([seed, tree_id, prefix]).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    choices = list(OPERATORS)
    rng.shuffle(choices)
    return choices


def analogy_preferences(problem: Problem) -> list[Counter]:
    votes = {}
    for tree in EXEMPLAR:
        shapes = _shapes(tree, tuple(OPERATORS))
        actions = [token for token in tree if token in OPERATORS]
        for shape, action in zip(shapes, actions):
            votes.setdefault(shape, Counter())[action] += 1
    return [votes.get(shape, Counter()) for tree in problem.trees for shape in _shapes(tree, ("?",))]


@dataclass(frozen=True)
class Node:
    id: int
    tree_id: int
    prefix: Prefix


def search(problem: Problem, method: str, *, caps: Caps = Caps(), seed: int = 17, roots: int = 3, proposer=None):
    """Compare single/multiple sampled paths and single/multiple DFS trees.

    Forests schedule independent trees round-robin. Each tree has a separate
    proposal seed and stack. All methods verify complete candidates in emission
    order, retain all successes, and select the first certified assignment.
    """
    if method not in METHODS or type(roots) is not int or roots < 1:
        raise ValueError("expected a known method and a positive root count")
    started = perf_counter()
    root_count = 1 if method in ("single_path", "tot") else roots
    fanout = 1 if method in ("single_path", "best_of_n") else len(OPERATORS)
    preferences = analogy_preferences(problem) if method == "fot_analogy" else None
    stacks = [[Node(i, i, ())] for i in range(root_count)]
    active = deque(range(root_count))
    next_id = root_count
    tokens = calls = expansions = model_input_tokens = 0
    evaluated = set()
    solutions = {}
    events = []
    leaves = []
    reasons = set()
    depth = problem.hole_count
    while active:
        if tokens == caps.tokens or expansions == caps.expansions:
            if tokens == caps.tokens:
                reasons.add("tokens")
            if expansions == caps.expansions:
                reasons.add("expansions")
            break
        tree_id = active.popleft()
        parent = stacks[tree_id].pop()
        if proposer is None:
            ranking = propose(seed, tree_id, parent.prefix)
        else:
            actions, input_tokens = proposer(problem, seed, tree_id, parent.prefix)
            ranking = list(actions)
            if len(ranking) != len(OPERATORS) or set(ranking) != OPERATORS.keys():
                raise ValueError("checkpoint proposer must rank all legal operators exactly once")
            if type(input_tokens) is not int or input_tokens < 1:
                raise ValueError("checkpoint proposer must report a positive model input token count")
            model_input_tokens += input_tokens
        if preferences is not None:
            votes = preferences[len(parent.prefix)]
            ranking.sort(key=lambda action: -votes[action])
        expansions += 1
        children = []
        event = {
            "parent": asdict(parent), "parent_depth": len(parent.prefix),
            "source_role": "visible_query_prefix", "target_role": "next_operator",
            "transitions": [], "unemitted_actions": 0,
        }
        events.append(event)
        for index, action in enumerate(ranking[:fanout]):
            if tokens == caps.tokens:
                event["unemitted_actions"] = fanout - index
                reasons.add("tokens")
                break
            child = Node(next_id, tree_id, parent.prefix + (action,))
            next_id += 1
            tokens += 1
            status = "open"
            if len(child.prefix) < depth:
                children.append(child)
            elif calls == caps.verifier_calls:
                reasons.add("verifier_calls")
                status = "unverified_budget"
            else:
                calls += 1
                evaluated.add(child.prefix)
                if verifies(problem, child.prefix):
                    status = "verified"
                    solutions.setdefault(child.prefix, child.id)
                else:
                    status = "rejected"
            event["transitions"].append({
                "proposed_action": action, "executed_child": asdict(child), "status": status,
            })
            if status != "open":
                leaves.append({"node": asdict(child), "status": status})
        # Stack order ensures the highest-ranked child is expanded first.
        stacks[tree_id].extend(reversed(children))
        if stacks[tree_id]:
            active.append(tree_id)

    for stack in stacks:
        for node in stack:
            leaves.append({"node": asdict(node), "status": "frontier_budget"})
    complete = len(evaluated) == len(OPERATORS) ** depth
    if len(solutions) > 1:
        verdict = "multiple_verified"
    elif solutions:
        verdict = "unique_proven" if complete else "one_verified"
    else:
        verdict = "infeasible_proven" if complete else "none_verified"
    return {
        "enumeration_complete": complete, "verdict": verdict,
        "method": method, "case": problem.name, "seed": seed, "roots": root_count,
        "caps": asdict(caps), "used": {"tokens": tokens, "expansions": expansions, "verifier_calls": calls},
        "elapsed_seconds": perf_counter() - started, "budget_exhausted": sorted(reasons),
        "model_calls": expansions if proposer is not None else 0,
        "model_input_tokens": model_input_tokens,
        "solutions": list(solutions), "selected_node": next(iter(solutions.values()), None),
        "selected": next(iter(solutions), None), "observed_ambiguity": len(solutions) > 1,
        "corrections": 0, "events": events, "leaves": leaves,
    }


def benchmark(*, caps: Caps = Caps(), seed: int = 17, roots: int = 3, traces: bool = False, proposer=None):
    results = []
    summary = {}
    for method in METHODS:
        runs = [search(problem, method, caps=caps, seed=seed, roots=roots, proposer=proposer) for problem in PROBLEMS]
        successes = sum(run["selected"] is not None for run in runs)
        summary[method] = {
            "verified_answer_rate": successes / len(runs),
            "solved": successes, "cases": len(runs),
            "unique_verified_solutions": sum(len(run["solutions"]) for run in runs),
            "exhausted_cases": sum(bool(run["budget_exhausted"]) for run in runs),
            "ambiguous_cases_observed": sum(run["observed_ambiguity"] for run in runs),
            **{key: sum(run["used"][key] for run in runs) for key in asdict(caps)},
            "elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
            "model_calls": sum(run["model_calls"] for run in runs),
            "model_input_tokens": sum(run["model_input_tokens"] for run in runs),
        }
        for run in runs:
            if not traces:
                del run["events"], run["leaves"]
            results.append(run)
    return {
        "fixture": FIXTURE_VERSION, "seed": seed, "caps_per_case": asdict(caps),
        "proposer": "seeded-shuffle-v1" if proposer is None else "native-checkpoint-v1",
        "action_unit": "one postfix operator token",
        "worked_example": EXEMPLAR, "problems": [asdict(p) for p in PROBLEMS],
        "summary": summary, "runs": results,
    }


# Ordered topology mode generates its own parent-child attachments. The source
# supplies ordered operands per root. Its original postfix grouping is hidden.
ADD_OPERAND = "ADD_CHILD operand"
CLOSE = "CLOSE"
ADD_OPERATORS = {f"ADD_CHILD {op}": op for op in OPERATORS}
TOPOLOGY_ACTIONS = (ADD_OPERAND, *ADD_OPERATORS, CLOSE)


@dataclass(frozen=True)
class OrderedPrefix:
    root: int = 0
    consumed: int = 0
    # Open ancestor labels and their unfilled child slots, root first.
    frames: tuple[tuple[str | int, int], ...] = ()

    @property
    def ancestors(self):
        return tuple(label for label, _ in self.frames)

    def legal(self, operands):
        if self.root == len(operands):
            return ()
        if self.frames and self.frames[-1][1] == 0:
            return (CLOSE,)
        leaves = len(operands[self.root]) - self.consumed
        slots = sum(n for _, n in self.frames) if self.frames else 1
        # Adding an operator increases open slots by one. Adding a leaf
        # cannot fill the last slot while more source operands remain.
        actions = [ADD_OPERAND] if slots > 1 or leaves == 1 else []
        if leaves > slots:
            actions.extend(ADD_OPERATORS)
        return tuple(actions)

    def advance(self, action, operands):
        if action not in self.legal(operands):
            raise ValueError(f"illegal ordered topology action {action!r}")
        frames = list(self.frames)
        if action == CLOSE:
            frames.pop()
            if not frames:
                return OrderedPrefix(self.root + 1)
            return OrderedPrefix(self.root, self.consumed, tuple(frames))
        if frames:
            parent, slots = frames[-1]
            frames[-1] = (parent, slots - 1)
        if action == ADD_OPERAND:
            frames.append((operands[self.root][self.consumed], 0))
            return OrderedPrefix(self.root, self.consumed + 1, tuple(frames))
        frames.append((ADD_OPERATORS[action], 2))
        return OrderedPrefix(self.root, self.consumed, tuple(frames))


def rank_topology(ancestors, *, seed):
    """Rank the entire action vocabulary before the grammar masks illegal moves."""
    material = json.dumps([seed, ancestors]).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    actions = list(TOPOLOGY_ACTIONS)
    rng.shuffle(actions)
    return actions


def verify_ordered_terminal(actions, operands, targets):
    """Independently decode a completion from its actions and evaluate each root."""
    forest = []
    prefix = iter(actions)
    end = object()

    # This parser uses no search state or legality mask. A CLOSE is required
    # for every node, and both operands of each operator must be present.
    def parse(literals):
        action = next(prefix)
        if action == ADD_OPERAND:
            value = next(literals)
            expression = value
        elif action in ADD_OPERATORS:
            left, x = parse(literals)
            right, y = parse(literals)
            op = ADD_OPERATORS[action]
            expression, value = (op, left, right), OPERATORS[op](x, y)
        else:
            raise ValueError("expected ADD_CHILD")
        if next(prefix) != CLOSE:
            raise ValueError("missing CLOSE")
        return expression, value

    if len(operands) != len(targets):
        return None
    try:
        for leaves, target in zip(operands, targets):
            literals = iter(leaves)
            expression, value = parse(literals)
            if value != target or next(literals, end) is not end:
                return None
            forest.append(expression)
    except (ValueError, StopIteration):
        return None
    return tuple(forest) if next(prefix, end) is end else None


def search_ordered_topology(problem, *, caps=Caps(), seed=17, traces=False, ranker=rank_topology):
    """Bounded depth-first search over all ordered full binary shapes and labels."""
    operands = tuple(tuple(x for x in tree if type(x) is int) for tree in problem.trees)
    frontier = [(OrderedPrefix(), ())]
    tokens = expansions = calls = 0
    reasons = set()
    solutions = []
    events = []
    while frontier:
        if tokens >= caps.tokens or expansions >= caps.expansions:
            if tokens >= caps.tokens:
                reasons.add("tokens")
            if expansions >= caps.expansions:
                reasons.add("expansions")
            break
        state, path = frontier.pop()
        expansions += 1
        ranking = tuple(ranker(state.ancestors, seed=seed))
        if len(ranking) != len(TOPOLOGY_ACTIONS) or set(ranking) != set(TOPOLOGY_ACTIONS):
            raise ValueError("topology ranker must return the entire action vocabulary once")
        legal = state.legal(operands)
        children = []
        for action in (a for a in ranking if a in legal):
            if tokens == caps.tokens:
                reasons.add("tokens")
                break
            child = state.advance(action, operands)
            trajectory = (*path, action)
            tokens += 1
            if child.root < len(operands):
                children.append((child, trajectory))
                status = "open"
            elif calls == caps.verifier_calls:
                reasons.add("verifier_calls")
                status = "unverified_budget"
            else:
                calls += 1
                forest = verify_ordered_terminal(trajectory, operands, problem.targets)
                status = "rejected" if forest is None else "verified"
                if forest is not None:
                    solutions.append(forest)
            if traces:
                events.append({"prefix": path, "ancestors": state.ancestors,
                               "action": action, "status": status})
        frontier.extend(reversed(children))
    return {
        "case": problem.name, "operands": operands, "targets": problem.targets,
        "complete_path_length": 2 * sum(2 * len(root) - 1 for root in operands),
        "used": {"tokens": tokens, "expansions": expansions, "verifier_calls": calls},
        "budget_exhausted": sorted(reasons), "solutions": solutions,
        "selected": solutions[0] if solutions else None,
        **({"events": events} if traces else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordered-topology", action="store_true",
                        help="Generate ordered arithmetic forest shapes with ADD_CHILD and CLOSE.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--roots", type=int, default=3)
    for name, value in asdict(Caps()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--trusted-checkpoint", help="Local native LM checkpoint with a pinned digest.")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    caps = Caps(args.tokens, args.expansions, args.verifier_calls)
    artifacts = (args.trusted_checkpoint, args.checkpoint_sha256, args.tokenizer, args.tokenizer_sha256)
    if args.ordered_topology:
        if any(artifacts):
            parser.error("ordered topology uses its own ancestor-only symbolic ranker")
        report = {
            "fixture": "ordered-topology-v1", "seed": args.seed, "caps_per_case": asdict(caps),
            "action_unit": "one ADD_CHILD or CLOSE", "method": "depth_first",
            "runs": [search_ordered_topology(p, caps=caps, seed=args.seed, traces=args.traces)
                     for p in PROBLEMS],
        }
        print(json.dumps(report, indent=2))
        return
    proposer = provenance = None
    if any(artifacts):
        if not all(artifacts):
            parser.error("checkpoint evaluation requires a trusted checkpoint, tokenizer and both SHA-256 pins")
        from toricgt.checkpoint_symbolic_search import checkpoint_proposer

        try:
            proposer, provenance = checkpoint_proposer(*artifacts, device=args.device)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    try:
        report = benchmark(seed=args.seed, caps=caps, roots=args.roots, traces=args.traces, proposer=proposer)
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if provenance is not None:
        report["checkpoint_provenance"] = provenance
        report["checkpoint_provenance"]["model_input_token_cap_per_case"] = (
            provenance["prompt_token_cap_per_call"] * caps.expansions
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
