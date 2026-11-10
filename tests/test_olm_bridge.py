import torch
from olm.models.alibaba.qwen2 import Qwen2Model
from olm.train.optim import AdamW
from olm.train.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import Qwen2Config, Qwen2ForCausalLM

from recipe_dapt.olm_model import (
    CausalNextTokenLoss,
    build_olm_qwen,
    hf_to_olm_state,
    olm_to_hf_state,
)


def tiny_models():
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    hf = Qwen2ForCausalLM(config)
    olm = Qwen2Model(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64,
        rope_theta=1_000_000.0,
        tie_weights=True,
        dropout=0.0,
        rms_norm_eps=1e-6,
    )
    return hf, olm


def test_hf_olm_roundtrip_and_forward_parity() -> None:
    torch.manual_seed(7)
    hf, olm = tiny_models()
    mapped = hf_to_olm_state(hf.state_dict(), num_layers=2)
    olm.load_state_dict(mapped, strict=True)
    roundtrip = olm_to_hf_state(olm.state_dict(), num_layers=2)
    hf.load_state_dict(roundtrip, strict=True)
    tokens = torch.randint(0, 128, (2, 17))
    with torch.no_grad():
        hf_logits = hf(tokens).logits
        olm_logits = olm(tokens)
    torch.testing.assert_close(hf_logits, olm_logits, atol=2e-3, rtol=2e-3)
    assert torch.equal(hf_logits.argmax(-1), olm_logits.argmax(-1))


def test_current_transformers_rope_config_schema() -> None:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
        tie_word_embeddings=True,
    )
    model = build_olm_qwen(config, dtype=torch.float32)
    assert model(torch.randint(0, 128, (1, 8))).shape == (1, 8, 128)


def test_qwen25_half_billion_full_shape_bridge() -> None:
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    with torch.device("meta"):
        hf = Qwen2ForCausalLM(config)
        olm = build_olm_qwen(config)
    assert sum(parameter.numel() for parameter in hf.parameters()) == 494_032_768
    assert sum(parameter.numel() for parameter in olm.parameters()) == 494_032_768

    olm_shapes = {key: value.shape for key, value in olm.state_dict().items()}
    converted_olm = hf_to_olm_state(hf.state_dict(), num_layers=24)
    assert {key: value.shape for key, value in converted_olm.items()} == olm_shapes

    hf_shapes = {key: value.shape for key, value in hf.state_dict().items()}
    converted_hf = olm_to_hf_state(converted_olm, num_layers=24)
    assert {key: value.shape for key, value in converted_hf.items()} == hf_shapes


def test_olm_trainer_runs_one_cpu_step() -> None:
    _, model = tiny_models()
    tokens = torch.randint(0, 128, (2, 16))
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=2)
    trainer = Trainer(
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-4),
        dataloader=loader,
        device="cpu",
        context_length=16,
        use_amp=False,
        loss=CausalNextTokenLoss,
        warmup_steps=1,
        total_steps=1,
    )
    losses = trainer.train(epochs=1, max_steps=1, log_interval=1)
    assert trainer.global_step == 1
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
