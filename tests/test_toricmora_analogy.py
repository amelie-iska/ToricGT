from collections import Counter
from dataclasses import replace

import pytest

from toricgt.toricmora_analogy import (
    AnalogyCase,
    IdentityWitness,
    TypedForest,
    anchor_payload_sizes,
    evaluate_analogies,
    forest_correspondence,
    generate_analogy_cases,
    generate_identity_analogy_cases,
    verify_analogy,
)


def _mapped_edges(edges, mapping):
    return {(mapping[parent], mapping[child], role) for parent, child, role in edges}


def _typed_edge_counts(forest):
    kinds = dict(forest.nodes)
    return Counter((kinds[parent], role, kinds[child]) for parent, child, role in forest.edges)


def test_correspondence_respects_direction_types_and_repeated_children():
    source = TypedForest(
        (("a", "root"), ("b", "tick"), ("c", "tick")),
        (("a", "b", "mark"), ("a", "c", "mark")),
    )
    target = TypedForest(
        (("z", "tick"), ("x", "root"), ("y", "tick")),
        (("x", "z", "mark"), ("x", "y", "mark")),
    )

    mapping = forest_correspondence(source, target)
    assert mapping is not None
    assert len(set(mapping.values())) == 3
    assert _mapped_edges(source.edges, mapping) == set(target.edges)
    reversed_edge = replace(target, edges=(("y", "x", "mark"), ("x", "z", "mark")))
    changed_role = replace(target, edges=(("x", "z", "other"), ("x", "y", "mark")))
    assert forest_correspondence(source, reversed_edge) is None
    assert forest_correspondence(source, changed_role) is None
    with pytest.raises(ValueError, match="cycles"):
        replace(source, edges=(("a", "b", "mark"), ("b", "a", "mark")))


def test_seeded_edge_swaps_preserve_marginals_and_certify_only_the_target():
    cases = generate_analogy_cases(seed=32, examples_per_scope=3)
    assert cases == generate_analogy_cases(seed=32, examples_per_scope=3)
    assert len({case.target_index for case in cases}) > 1

    for case in cases:
        verified = verify_analogy(case)
        assert verified.keys() == {case.target_index}
        witness = verified[case.target_index]
        target = case.candidates[case.target_index]
        query_edges = set(case.query.edges)
        assert set(witness.removed_query_edges) <= query_edges
        assert set(witness.added_query_edges).isdisjoint(query_edges)
        transported = (query_edges - set(witness.removed_query_edges)) | set(witness.added_query_edges)
        assert _mapped_edges(transported, witness.query_to_candidate_nodes) == set(target.edges)
        mapped_types = {witness.query_to_candidate_nodes[node]: kind for node, kind in case.query.nodes}
        assert mapped_types == dict(target.nodes)
        assert len(witness.source_to_query_branches) == 4
        assert len(case.query.roots) == (5 if case.scope == "forest" else 1)
        assert dict(case.source.nodes).keys().isdisjoint(dict(case.source_result.nodes))
        for candidate in case.candidates:
            assert dict(case.query.nodes).keys().isdisjoint(dict(candidate.nodes))
            candidate_types = Counter(kind for _, kind in candidate.nodes)
            query_types = Counter(kind for _, kind in case.query.nodes)
            assert candidate_types == query_types
            assert _typed_edge_counts(candidate) == _typed_edge_counts(case.query)


def test_bench_reports_scopes_and_expected_top1_credit_for_ties():
    cases = generate_analogy_cases(examples_per_scope=2)
    results = evaluate_analogies(cases)
    assert set(results) == {"all", "subtree", "tree", "forest"}
    assert results["all"]["cases"] == 6
    for report in results.values():
        assert report["offset_top1_expected"] == pytest.approx(1 / 6)
        assert report["offset_tie_rate"] == 1
        assert report["structural_target_recall"] == 1
        assert report["structural_false_accept_rate"] == 0

    assert evaluate_analogies(cases, embed=lambda _: (1.0, 2.0)) == results
    with pytest.raises(ValueError, match="finite"):
        evaluate_analogies(cases, embed=lambda _: (float("nan"),))


def test_verifier_rejects_missing_source_exchange_and_reports_false_accepts():
    case = generate_analogy_cases(examples_per_scope=1)[0]
    assert verify_analogy(replace(case, source_result=case.source)) == {}
    target = case.candidates[case.target_index]
    duplicate = replace(case, candidates=(*case.candidates, target))
    report = evaluate_analogies([duplicate])["all"]
    assert report["structural_false_accept_rate"] == pytest.approx(1 / 6)
    assert report["structural_target_recall"] == 1
    assert report["structural_unique_top1"] == 0


def test_anchor_payload_size_embedding_preserves_shapes_and_sees_edge_swaps():
    case = generate_analogy_cases(examples_per_scope=1)[0]
    input_sizes = anchor_payload_sizes(case.query)
    assert sorted(input_sizes) == [4, 5, 6, 7]
    candidate_sizes = {anchor_payload_sizes(candidate) for candidate in case.candidates}
    assert len(candidate_sizes) == 6
    assert all(sorted(sizes) == sorted(input_sizes) for sizes in candidate_sizes)
    assert all(sizes != input_sizes for sizes in candidate_sizes)


def test_untouched_same_anchor_subtree_does_not_shadow_selected_exchange():
    case = generate_analogy_cases(examples_per_scope=1)[0]
    _, body, _ = next(edge for edge in case.query.edges if edge[2] == "body")
    descendants = [body]
    for node in descendants:
        descendants.extend(child for parent, child, _ in case.query.edges if parent == node)
    clone = {node: f"decoy-{node}" for node in descendants}
    clone_nodes = tuple((clone[node], kind) for node, kind in case.query.nodes if node in clone)
    clone_edges = tuple(
        (clone[parent], clone[child], role)
        for parent, child, role in case.query.edges
        if parent in clone
    )

    def with_decoy(forest):
        return TypedForest(
            forest.nodes + clone_nodes,
            forest.edges + clone_edges + ((forest.roots[0], clone[body], "side"),),
        )

    query = with_decoy(case.query)
    candidates = tuple(with_decoy(candidate) for candidate in case.candidates)
    assert verify_analogy(replace(case, query=query, candidates=candidates)).keys() == {
        case.target_index
    }


def test_right_identity_requires_the_exemplar_and_respects_scope():
    cases = generate_identity_analogy_cases(seed=31, examples_per_scope=2)
    assert cases == generate_identity_analogy_cases(seed=31, examples_per_scope=2)
    assert len(cases) == 18
    for index in range(0, len(cases), 3):
        group = cases[index:index + 3]
        first = group[0]
        assert all(case.source == first.source and case.query == first.query for case in group)
        assert all(case.candidates == first.candidates for case in group)
        assert len({case.target_index for case in group}) == 3
        operators = set()
        for case in group:
            accepted = verify_analogy(case)
            assert accepted.keys() == {case.target_index}
            witness = accepted[case.target_index]
            assert isinstance(witness, IdentityWitness)
            operators.add(witness.operator)
            assert len(witness.removed_query_sites) == (2 if case.scope == "forest" else 1)
        assert operators == {"add", "sub", "mul"}

    assert verify_analogy(replace(cases[0], source_result=cases[0].source)) == {}
    for report in evaluate_analogies(cases).values():
        assert report["structural_unique_top1"] == 1
        assert report["structural_false_accept_rate"] == 0
        assert report["unshifted_top1_expected"] <= 1 / 3 + 1e-12


def test_identity_rewrite_uses_original_sites_instead_of_repeating_to_fixed_point():
    def nested(symbol):
        return TypedForest(
            (("outer", "add"), ("symbol", symbol), ("inner", "add"), ("a", "0"), ("b", "0")),
            (
                ("outer", "symbol", "lhs"),
                ("outer", "inner", "rhs"),
                ("inner", "a", "lhs"),
                ("inner", "b", "rhs"),
            ),
        )

    def single(symbol):
        return TypedForest(
            (("outer", "add"), ("symbol", symbol), ("zero", "0")),
            (("outer", "symbol", "lhs"), ("outer", "zero", "rhs")),
        )

    case = AnalogyCase(
        "tree",
        nested("x"),
        single("x"),
        nested("y"),
        (single("y"), TypedForest((("symbol", "y"),), ())),
        0,
        "right-identity",
    )
    accepted = verify_analogy(case)
    assert accepted.keys() == {0}
    assert isinstance(accepted[0], IdentityWitness)
    assert accepted[0].removed_query_sites == ("inner",)


def test_relational_distortion_isometry_vs_shift():
    from toricgt.toricmora_analogy import relational_distortion
    # 3 points forming a right-angled triangle
    d_source = [
        [0.0, 3.0, 4.0],
        [3.0, 0.0, 5.0],
        [4.0, 5.0, 0.0],
    ]
    # Isomorphic target rotated/reflected with same distances
    d_target_isometric = [
        [0.0, 3.0, 4.0],
        [3.0, 0.0, 5.0],
        [4.0, 5.0, 0.0],
    ]
    # Distorted target where distances change
    d_target_distorted = [
        [0.0, 2.0, 6.0],
        [2.0, 0.0, 4.0],
        [6.0, 4.0, 0.0],
    ]
    corr = ((0, 0), (1, 1), (2, 2))
    assert relational_distortion(d_source, d_target_isometric, corr) == 0.0
    assert relational_distortion(d_source, d_target_distorted, corr) > 0.0


def test_multi_root_forest_consensus_combines_majority_trees():
    from toricgt.toricmora_analogy import TypedForest, multi_root_forest_consensus

    tree_alpha = (
        (("r1", "compute"), ("c1", "op"), ("c2", "leaf")),
        (("r1", "c1", "body"), ("c1", "c2", "arg")),
    )
    tree_beta = (
        (("r2", "routine"), ("s1", "step")),
        (("r2", "s1", "next"),),
    )
    tree_gamma = (
        (("r3", "alt"), ("a1", "arg")),
        (("r3", "a1", "in"),),
    )

    f1 = TypedForest(tree_alpha[0] + tree_beta[0], tree_alpha[1] + tree_beta[1])
    f2 = TypedForest(tree_alpha[0] + tree_gamma[0], tree_alpha[1] + tree_gamma[1])
    f3 = TypedForest(tree_alpha[0] + tree_beta[0], tree_alpha[1] + tree_beta[1])

    # Consensus with min_agreement 0.6 should retain tree_alpha (3/3) and tree_beta (2/3), but reject tree_gamma (1/3)
    consensus = multi_root_forest_consensus([f1, f2, f3], min_agreement=0.6)
    assert len(consensus.roots) == 2


def test_consensus_counts_each_forest_once():
    from toricgt.toricmora_analogy import multi_root_forest_consensus

    duplicate_vote = TypedForest((("a", "minority"), ("b", "minority"), ("c", "minority")), ())
    first = TypedForest((("p", "shared"),), ())
    second = TypedForest((("q", "shared"),), ())
    for forests in ([duplicate_vote, first, second], [second, duplicate_vote, first]):
        consensus = multi_root_forest_consensus(forests, min_agreement=0.6)
        assert [kind for _, kind in consensus.nodes] == ["shared"]
