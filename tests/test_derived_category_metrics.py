import torch

from toricgt.derived_category_metrics import (
    DerivedCategoryConfig,
    analogical_derived_category_loss,
    chain_complex_from_edges_np,
    derived_category_feature_summary,
    derived_category_objects_from_batch,
    projective_resolution_certificate,
)


def test_branch_merge_chain_complex_has_nontrivial_cycle():
    edges = [[0, 1], [0, 2], [1, 3], [2, 3]]

    complex_object = chain_complex_from_edges_np(4, edges, max_vertices=8)

    assert complex_object["kind"] == "got_simplicial_chain_complex"
    assert complex_object["num_vertices"] == 4
    assert len(complex_object["edges"]) == 4
    assert complex_object["boundary_1"]
    assert complex_object["betti"]["beta_0"] == 1
    assert complex_object["betti"]["beta_1"] == 1
    assert complex_object["euler_characteristic"] == 0


def test_projective_resolution_certificate_contains_symbolic_dg_data():
    certificate = projective_resolution_certificate(5)
    resolution = certificate["resolution"]

    assert certificate["ambient_category"] == "D^b(grmod-S)"
    assert resolution["ring"]["kind"] == "multigraded_polynomial_ring"
    assert resolution["stanley_reisner_ideal"]["generator_count"] > 0
    assert len(resolution["hochster_betti_rows"]) > 0
    assert len(resolution["dg_differential_entries"]) > 0
    assert len(resolution["fitting_entry_ideal_rows"]) > 0
    assert len(resolution["fitting_summary_rows"]) > 0
    assert len(resolution["dg_product_summary_rows"]) > 0


def test_analogical_derived_category_loss_is_finite_for_branch_merge_batch():
    hidden = torch.randn(4, 8, 32)
    node_mask = torch.ones(4, 8, dtype=torch.bool)
    edge_index = torch.tensor(
        [[[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]] * 4
    )
    edge_mask = torch.ones(4, edge_index.shape[1], dtype=torch.bool)

    metrics = analogical_derived_category_loss(
        hidden,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        config=DerivedCategoryConfig(max_vertices=8),
    )

    assert metrics["derived_category_loss"].isfinite()
    assert metrics["derived_category_chain_map_residual"].isfinite()
    assert metrics["derived_category_mapping_cone_residual"].isfinite()
    assert metrics["derived_category_projective_resolution_distance"].isfinite()
    assert metrics["derived_category_exact_projective_dimension"] > 0
    assert metrics["derived_category_pairs"] == 4


def test_derived_category_feature_summary_and_batch_objects():
    hidden = torch.randn(2, 6, 16)
    node_mask = torch.ones(2, 6, dtype=torch.bool)
    edge_index = torch.tensor([[[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]]] * 2)
    edge_mask = torch.ones(2, edge_index.shape[1], dtype=torch.bool)

    features = derived_category_feature_summary(
        hidden,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
    )
    objects = derived_category_objects_from_batch(edge_index, node_mask=node_mask, edge_mask=edge_mask)

    assert features.shape == (2, 11)
    assert torch.isfinite(features).all()
    assert objects[0]["chain_complex"]["kind"] == "got_simplicial_chain_complex"
    assert objects[0]["projective_resolution"]["resolution"]["stanley_reisner_ideal"]["generator_count"] > 0
