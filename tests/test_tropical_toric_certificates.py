from __future__ import annotations

import pytest

from toricgt.tropical_toric_certificates import tropical_hypersurface_certificate


def test_triangle_tropical_hypersurface_is_exactly_balanced() -> None:
    cert = tropical_hypersurface_certificate([[0, 0], [1, 0], [0, 1]], [0, 0, 0])

    assert cert["kind"] == "closed_form_tropical_hypersurface_multiplicity_balance_certificate"
    tropical = cert["tropical"]
    assert tropical["multiplicities"] == [1, 1, 1]
    assert tropical["balanced"] is True
    assert tropical["max_balance_residual_l1"] == 0
    assert tropical["chow_minkowski_weight_certified"] is True
    assert len(tropical["facets"]) == 3
    assert len(tropical["balancing_stars"]) == 1
    assert tropical["balancing_stars"][0]["balance_residual"] == [0, 0]
    assert cert["toric"]["chow_class_certified"] is True


def test_tropical_hypersurface_uses_lattice_gcd_multiplicities() -> None:
    cert = tropical_hypersurface_certificate([[0, 0], [2, 0], [0, 2]], [0, 0, 0])
    weights = sorted(facet["multiplicity"] for facet in cert["tropical"]["facets"])

    assert weights == [2, 2, 2]
    assert cert["tropical"]["balanced"] is True


def test_tropical_hypersurface_rejects_non_2d_support() -> None:
    with pytest.raises(ValueError, match="exponent_dim == 2"):
        tropical_hypersurface_certificate([[0, 0, 0], [1, 0, 0], [0, 1, 1]], [0, 0, 0])
