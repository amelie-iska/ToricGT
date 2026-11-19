"""Exact finite-DAG oracle for discrete trajectory balance on symbolic fixtures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from graphlib import TopologicalSorter
from math import log
from statistics import fmean

STOP = "STOP"
Step = tuple[str, str, str]  # source, action, successor
Path = tuple[Step, ...]


def _positive_rational(value: Fraction | int) -> Fraction:
    if type(value) not in (int, Fraction) or value <= 0:
        raise ValueError("expected a positive integer or Fraction")
    return Fraction(value)


@dataclass(frozen=True)
class Transition:
    source: str
    action: str
    successor: str
    forward: Fraction
    backward: Fraction

    @property
    def step(self) -> Step:
        return self.source, self.action, self.successor


@dataclass(frozen=True)
class Balance:
    terminal: str
    forward_mass: Fraction
    backward_mass: Fraction
    # Z times forward path probability divided by reward times backward path probability.
    ratio: Fraction

    @property
    def squared_log_residual(self) -> float:
        return (log(self.ratio.numerator) - log(self.ratio.denominator)) ** 2


class SymbolicDAG:
    """One root and explicit STOP edges into terminal reward states.

    Each legal edge has positive rational forward and reverse probability.
    Forward probabilities normalize over edges leaving a state. Reverse
    probabilities normalize over edges entering each successor, including
    terminal states. Each action from a given source has a unique successor.
    """

    def __init__(self, root: str, edges: tuple[Transition, ...], rewards: dict[str, Fraction]):
        self.root = root
        self._rewards = {state: _positive_rational(reward) for state, reward in rewards.items()}
        if not self._rewards or root in self._rewards:
            raise ValueError("expected non-root terminal rewards")
        self._outgoing: dict[str, dict[str, Transition]] = defaultdict(dict)
        incoming = defaultdict(list)
        states = {root, *rewards}
        for edge in edges:
            _positive_rational(edge.forward)
            _positive_rational(edge.backward)
            if edge.source in rewards or edge.successor == root:
                raise ValueError("terminal states cannot emit edges and the root cannot receive edges")
            if (edge.action == STOP) != (edge.successor in rewards):
                raise ValueError("only STOP actions lead to terminal states")
            if edge.action in self._outgoing[edge.source]:
                raise ValueError("duplicate action from one source")
            self._outgoing[edge.source][edge.action] = edge
            incoming[edge.successor].append(edge)
            states.update((edge.source, edge.successor))

        parents = {s: {edge.source for edge in incoming[s]} for s in states}
        tuple(TopologicalSorter(parents).static_order())  # raises CycleError on cycles
        for state in states:
            if state not in rewards and sum(e.forward for e in self._outgoing[state].values()) != 1:
                raise ValueError(f"forward probabilities must sum to one at {state}")
            if state != root and sum(e.backward for e in incoming[state]) != 1:
                raise ValueError(f"backward probabilities must sum to one at successor {state}")

    def paths(self):
        """Enumerate every complete root-to-terminal path for small fixtures."""
        pending = [(self.root, ())]
        while pending:
            state, prefix = pending.pop()
            if state in self._rewards:
                yield prefix
            else:
                for edge in reversed(self._outgoing[state].values()):
                    pending.append((edge.successor, (*prefix, edge.step)))

    def balance(self, path: Path, partition: Fraction | int) -> Balance:
        state = self.root
        forward = backward = Fraction(1)
        z = _positive_rational(partition)
        for source, action, successor in path:
            if state != source or state in self._rewards:
                raise ValueError("trajectory must follow connected edges from the root and end on STOP")
            edge = self._outgoing[state].get(action)
            if edge is None or edge.successor != successor:
                raise ValueError("illegal action or successor")
            forward *= edge.forward
            backward *= edge.backward
            state = successor
        if state not in self._rewards:
            raise ValueError("trajectory needs an explicit STOP transition")
        return Balance(state, forward, backward, z * forward / (self._rewards[state] * backward))

    def audit(self, partition: Fraction | int):
        """Exact pathwise checks and marginals, plus a uniform-over-paths log-TB loss."""
        balances = [self.balance(path, partition) for path in self.paths()]
        marginal = dict.fromkeys(self._rewards, Fraction(0))
        for balance in balances:
            marginal[balance.terminal] += balance.forward_mass
        total_reward = sum(self._rewards.values())
        target = {s: reward / total_reward for s, reward in self._rewards.items()}
        return {
            "paths": len(balances),
            "balanced_paths": sum(b.ratio == 1 for b in balances),
            "terminal_mass": marginal,
            "normalized_rewards": target,
            "matches_rewards": marginal == target,
            "mean_squared_log_residual": fmean(b.squared_log_residual for b in balances),
        }


def subset_fixture() -> SymbolicDAG:
    """Build an ordered pair of additions with a merge at the set {a,b}."""
    return SymbolicDAG(
        "empty",
        (
            Transition("empty", "add-a", "a", Fraction(4, 7), Fraction(1)),
            Transition("empty", "add-b", "b", Fraction(3, 7), Fraction(1)),
            Transition("a", STOP, "done-a", Fraction(1, 2), Fraction(1)),
            Transition("a", "add-b", "ab", Fraction(1, 2), Fraction(2, 5)),
            Transition("b", "add-a", "ab", Fraction(1), Fraction(3, 5)),
            Transition("ab", STOP, "done-ab", Fraction(1), Fraction(1)),
        ),
        {"done-a": Fraction(2), "done-ab": Fraction(5)},
    )
