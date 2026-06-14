import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from toricgt.cas_backed_losses import (
    balanced_cycle_loss_from_certificate,
    cartier_bend_loss_from_certificate,
    toric_binomial_loss_from_certificate,
)
from toricgt.cas_certificates import CertificateCache, validate_certificate_payload
from toricgt.cas_oracles import (
    CASUnavailableError,
    Macaulay2TropicalOracle,
    SageToricOracle,
    cyclic_stanley_reisner_closed_form_certificate,
    discover_all_backends,
)


def test_closed_form_cyclic_certificate_is_exact_and_valid(tmp_path: Path) -> None:
    cert = cyclic_stanley_reisner_closed_form_certificate(6)
    payload = cert.with_hash()

    assert payload["provenance"] == "exact_closed_form"
    assert payload["tropical"]["balanced"] is True
    assert payload["commutative_algebra"]["dg_algebra"]["d_squared_zero"] is True
    assert payload["commutative_algebra"]["dg_algebra"]["leibniz_rule"] is True
    assert payload["diagnostics"]["symbolic_resolution_minimal_total_betti"] > 0
    assert validate_certificate_payload(payload) == []

    cache = CertificateCache(tmp_path / "cache")
    path = cache.write(cert)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["certificate_hash"] == cert.certificate_hash
    assert validate_certificate_payload(loaded) == []
    assert cache.read_hash(cert.certificate_hash)["kind"] == cert.kind
    assert cache.find_by_input_hash(cert.input_hash, kind=cert.kind) == [path]


def test_validate_rejects_surrogate_certificate() -> None:
    payload = {
        "kind": "bad_surrogate",
        "input_hash": "abc",
        "provenance": "surrogate_torch",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": {},
        "cas": {},
    }

    errors = validate_certificate_payload(payload)
    assert any("surrogate_torch" in error for error in errors)


def test_backend_discovery_is_explicit() -> None:
    statuses = discover_all_backends()

    assert set(statuses) == {"sage", "macaulay2"}
    for info in statuses.values():
        assert info.provenance in {"exact_cas/sage", "exact_cas/macaulay2", "cas_unavailable"}
        if not info.available:
            assert info.error


def test_sage_oracle_raises_when_unavailable() -> None:
    oracle = SageToricOracle(executable=None)
    if oracle.info.available:
        pytest.skip("Sage is installed; unavailable-path check is not applicable")
    with pytest.raises(CASUnavailableError):
        oracle.normal_fan_certificate([[0, 0], [1, 0], [0, 1]])


def test_macaulay2_oracle_raises_when_unavailable() -> None:
    oracle = Macaulay2TropicalOracle(executable=None)
    if oracle.info.available:
        pytest.skip("Macaulay2 is installed; unavailable-path check is not applicable")
    with pytest.raises(CASUnavailableError):
        oracle.smoke_certificate()


def test_cas_backed_binomial_loss_uses_exact_certificate_relations() -> None:
    certificate = {
        "kind": "unit_exact_relations",
        "input_hash": "unit",
        "provenance": "exact_cas/macaulay2",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": {},
        "cas": {},
        "commutative_algebra": {
            "toric_ideal_relations": [[0, 1, 2, 3]],
        },
    }
    logits = torch.tensor([[[1.0, 2.0, 1.5, 1.5], [3.0, 5.0, 4.0, 4.0]]], requires_grad=True)

    loss = toric_binomial_loss_from_certificate(logits, certificate, normalize=False)

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_balanced_cycle_and_cartier_losses_use_exact_certificate_targets() -> None:
    certificate = {
        "kind": "unit_exact_toric_tropical",
        "input_hash": "unit",
        "provenance": "exact_cas/sage",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": {},
        "cas": {},
        "tropical": {
            "balancing_stars": [
                {
                    "facets": [0, 1, 2],
                    "primitive_normals": [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
                }
            ]
        },
        "toric": {
            "cartier_wall_bends": [
                {"left": 0, "right": 1, "normal": [1.0, 0.0], "bend": 2.0},
            ]
        },
    }
    multiplicities = torch.ones(3, requires_grad=True)
    slopes = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)

    balance = balanced_cycle_loss_from_certificate(multiplicities, certificate)
    bend = cartier_bend_loss_from_certificate(slopes, certificate)

    assert balance.item() == pytest.approx(0.0)
    assert bend.item() == pytest.approx(0.0)
    (balance + bend).backward()
    assert multiplicities.grad is not None
    assert slopes.grad is not None


def test_build_and_validate_scripts_round_trip_closed_form(tmp_path: Path) -> None:
    out = tmp_path / "certs"
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_toric_tropical_certificates.py",
            "--output-dir",
            str(out),
            "--num-vertices",
            "6",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert build.returncode == 0, build.stdout
    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert report["certificates"]
    assert report["certificates"][0]["provenance"] == "exact_closed_form"

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/validate_cas_certificates.py",
            str(out / "cache"),
            "--allow-unavailable",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validate.returncode == 0, validate.stdout
    validation = json.loads(validate.stdout)
    assert validation["failed"] == 0


def test_exact_sage_normal_fan_if_installed() -> None:
    oracle = SageToricOracle()
    if not oracle.info.available:
        pytest.skip("SageMath not installed")
    cert = oracle.normal_fan_certificate([[0, 0], [1, 0], [0, 1], [1, 1]])
    payload = cert.with_hash()
    assert payload["provenance"] == "exact_cas/sage"
    assert payload["toric"]["num_rays"] > 0
    assert validate_certificate_payload(payload) == []


def test_exact_macaulay2_smoke_if_installed() -> None:
    oracle = Macaulay2TropicalOracle()
    if not oracle.info.available:
        pytest.skip("Macaulay2 not installed")
    cert = oracle.smoke_certificate()
    payload = cert.with_hash()
    assert payload["provenance"] == "exact_cas/macaulay2"
    assert payload["commutative_algebra"]["ring_ok"] is True
    assert validate_certificate_payload(payload) == []
