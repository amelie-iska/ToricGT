from __future__ import annotations

import torch

from toricgt.structure_flow_matching import (
    contact_bce_from_logits,
    contact_map,
    distogram_cross_entropy,
    pairwise_distances,
    structure_flow_loss,
)


def test_pairwise_distances_and_contacts_respect_mask() -> None:
    coords = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [10.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    dist = pairwise_distances(coords, mask)
    assert torch.isclose(dist[0, 0, 1], torch.tensor(5.0))
    assert dist[0, 0, 2].item() == 0.0
    contacts = contact_map(coords, mask, cutoff=6.0)
    assert contacts[0, 0, 1].item() == 1.0
    assert contacts[0, 0, 0].item() == 0.0
    assert contacts[0, 1, 2].item() == 0.0


def test_contact_and_distogram_losses_are_finite() -> None:
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True, True]])
    contact_logits = torch.zeros(1, 4, 4)
    contact_loss = contact_bce_from_logits(contact_logits, coords, mask, cutoff=4.0)
    assert torch.isfinite(contact_loss)
    dist_logits = torch.zeros(1, 4, 4, 5)
    bin_edges = torch.tensor([0.0, 2.5, 5.0, 8.0, 12.0, 20.0])
    dist_loss = distogram_cross_entropy(dist_logits, coords, mask, bin_edges)
    assert torch.isfinite(dist_loss)


def test_structure_flow_loss_recovers_zero_velocity_target() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]])
    noisy = target - 0.25
    pred_velocity = target - noisy
    mask = torch.tensor([[True, True]])
    metrics = structure_flow_loss(
        pred_velocity=pred_velocity,
        noisy_coords=noisy,
        target_coords=target,
        mask=mask,
        flow_weight=1.0,
    )
    assert metrics.flow_matching_loss.item() == 0.0
    assert metrics.rmsd.item() == 0.0
    assert metrics.coordinate_count.item() == 2.0
