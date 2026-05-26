"""Cyclic and braided Soft-MoE expert curricula."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertCurriculumAssignment:
    """One optimizer-step assignment for cyclic expert training."""

    phase: int
    round_index: int
    active_expert: int
    subset_id: int
    full_coverage_cycles: int
    full_coverage_complete: bool
    teacher_expert: int | None


class CyclicExpertCurriculum:
    """Assign experts to disjoint data subsets in cyclic or braided order.

    For ``num_experts == num_subsets``, phase zero sends expert ``e`` to subset
    ``e``.  After every expert has had ``phase_steps`` optimizer steps, the
    round increments.  After ``num_subsets`` rounds, every expert has seen every
    subset exactly once.  The braided order uses stride ``num_subsets - 1``,
    which reverses the cycle and is coprime to ``num_subsets``; this gives every
    expert a different orbit while preserving full coverage.
    """

    def __init__(
        self,
        num_experts: int,
        num_subsets: int,
        phase_steps: int,
        order: str = "braid",
        start_step: int = 0,
    ) -> None:
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if num_subsets < 1:
            raise ValueError("num_subsets must be positive")
        if phase_steps < 1:
            raise ValueError("phase_steps must be positive")
        if order not in {"cyclic", "braid"}:
            raise ValueError("order must be 'cyclic' or 'braid'")
        self.num_experts = num_experts
        self.num_subsets = num_subsets
        self.phase_steps = phase_steps
        self.order = order
        self.start_step = start_step
        self.stride = 1 if order == "cyclic" else max(1, num_subsets - 1)

    def assignment(self, step: int) -> ExpertCurriculumAssignment:
        local_step = max(0, step - self.start_step)
        phase = local_step // self.phase_steps
        active_expert = phase % self.num_experts
        round_index = phase // self.num_experts
        subset_id = (active_expert + self.stride * round_index) % self.num_subsets
        full_coverage_cycles = round_index // self.num_subsets
        full_coverage_complete = round_index >= self.num_subsets
        teacher_expert = None
        if full_coverage_complete and self.num_experts > 1:
            # Distill from a different expert that encountered the same subset
            # at a different recency.  The offset changes every full coverage
            # cycle so no single expert remains the permanent teacher.
            offset = 1 + (full_coverage_cycles % (self.num_experts - 1))
            teacher_expert = (active_expert - offset) % self.num_experts
        return ExpertCurriculumAssignment(
            phase=phase,
            round_index=round_index,
            active_expert=active_expert,
            subset_id=subset_id,
            full_coverage_cycles=full_coverage_cycles,
            full_coverage_complete=full_coverage_complete,
            teacher_expert=teacher_expert,
        )

    def expert_order(self, expert_idx: int) -> list[int]:
        if not (0 <= expert_idx < self.num_experts):
            raise IndexError("expert index out of range")
        return [(expert_idx + self.stride * round_idx) % self.num_subsets for round_idx in range(self.num_subsets)]
