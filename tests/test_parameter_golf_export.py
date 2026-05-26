from toricgt.config import ModelConfig
from toricgt.model import ToricTokenGT
from toricgt.parameter_golf_export import PARAMETER_GOLF_BYTE_LIMIT, write_artifact
from toricgt.random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig


def test_parameter_golf_artifact_report(tmp_path):
    cfg = ModelConfig(d_model=32, num_heads=4, num_layers=1, max_nodes=8, max_edges=16)
    model = ToricTokenGT(cfg)
    report = write_artifact(model, tmp_path / "artifact.zip", config=cfg.__dict__, bits=8)
    assert report.bytes_total > 0
    assert report.bytes_limit == PARAMETER_GOLF_BYTE_LIMIT
    assert report.within_limit
    assert (tmp_path / "artifact.zip.json").exists()


def test_random_order_dense_lm_artifact_report(tmp_path):
    cfg = RandomOrderLMConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=32,
        num_heads=4,
        num_layers=1,
        recurrent_passes=1,
        ffn_multiplier=2,
        dropout=0.0,
        polarquant_kv_bits=0,
    )
    model = DenseRandomOrderToricLM(cfg)
    report = write_artifact(model, tmp_path / "random_order_artifact.zip", config=cfg.__dict__, bits=8)
    assert report.bytes_total > 0
    assert report.within_limit
