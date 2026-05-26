from toricgt.config import ModelConfig
from toricgt.model import ToricTokenGT
from toricgt.parameter_golf_export import PARAMETER_GOLF_BYTE_LIMIT, write_artifact


def test_parameter_golf_artifact_report(tmp_path):
    cfg = ModelConfig(d_model=32, num_heads=4, num_layers=1, max_nodes=8, max_edges=16)
    model = ToricTokenGT(cfg)
    report = write_artifact(model, tmp_path / "artifact.zip", config=cfg.__dict__, bits=8)
    assert report.bytes_total > 0
    assert report.bytes_limit == PARAMETER_GOLF_BYTE_LIMIT
    assert report.within_limit
    assert (tmp_path / "artifact.zip.json").exists()
