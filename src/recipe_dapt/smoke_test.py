from __future__ import annotations

import argparse

import torch
from olm.models.alibaba.qwen2 import Qwen2Model
from olm.train.optim import AdamW
from olm.train.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import Qwen2Config, Qwen2ForCausalLM

from .olm_model import CausalNextTokenLoss, hf_to_olm_state, olm_to_hf_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the pinned OLM integration")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check checkpoint compatibility without running an optimizer step",
    )
    args = parser.parse_args()

    torch.manual_seed(19)
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
    hf = Qwen2ForCausalLM(config).eval()
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
    ).eval()
    olm.load_state_dict(hf_to_olm_state(hf.state_dict(), 2), strict=True)
    hf.load_state_dict(olm_to_hf_state(olm.state_dict(), 2), strict=True)
    tokens = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        hf_logits = hf(tokens).logits
        olm_logits = olm(tokens)
    torch.testing.assert_close(hf_logits, olm_logits, atol=2e-3, rtol=2e-3)
    if not torch.equal(hf_logits.argmax(-1), olm_logits.argmax(-1)):
        raise RuntimeError("HF/OLM bridge changed next-token predictions")

    if args.quick:
        print("Pinned OLM checkpoint bridge check passed.")
        return

    olm.train()
    trainer = Trainer(
        model=olm,
        optimizer=AdamW(olm.parameters(), lr=1e-4),
        dataloader=DataLoader(TensorDataset(tokens, tokens), batch_size=2),
        device="cpu",
        context_length=16,
        use_amp=False,
        loss=CausalNextTokenLoss,
        warmup_steps=1,
        total_steps=1,
    )
    losses = trainer.train(epochs=1, max_steps=1, log_interval=1)
    if trainer.global_step != 1 or len(losses) != 1:
        raise RuntimeError("OLM trainer did not complete the one-step smoke test")
    print("OLM bridge and one-step CPU training smoke test passed.")


if __name__ == "__main__":
    main()
