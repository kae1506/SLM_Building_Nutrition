from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
from olm.models.alibaba.qwen2 import Qwen2Model
from olm.train.losses.base import LossBase
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


OLM_GIT_COMMIT = "eeab195fcd3447fd5d6c39b59ca9224fb66b3ac7"


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class CausalNextTokenLoss(LossBase):
    """Causal LM loss for OLM, which otherwise compares unshifted positions."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )


def validate_qwen_config(config: Any) -> None:
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError(
            f"OLM bridge supports Qwen2/Qwen2.5 checkpoints, got {config.model_type!r}"
        )
    if bool(getattr(config, "use_sliding_window", False)):
        raise ValueError("OLM Qwen2 bridge does not support sliding-window checkpoints")
    if int(config.hidden_size) % int(config.num_attention_heads):
        raise ValueError("Qwen hidden_size must be divisible by num_attention_heads")
    expected_head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    configured_head_dim = int(getattr(config, "head_dim", expected_head_dim))
    if configured_head_dim != expected_head_dim:
        raise ValueError("OLM Qwen2 requires head_dim == hidden_size / num_attention_heads")


def qwen_rope_theta(config: Any) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    legacy_value = getattr(config, "rope_theta", None)
    if legacy_value is not None:
        return float(legacy_value)
    raise ValueError("Qwen config does not define rope_parameters.rope_theta")


def build_olm_qwen(config: Any, dtype: torch.dtype = torch.bfloat16) -> Qwen2Model:
    validate_qwen_config(config)
    with default_dtype(dtype):
        return Qwen2Model(
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_layers=int(config.num_hidden_layers),
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            max_seq_len=int(config.max_position_embeddings),
            rope_theta=qwen_rope_theta(config),
            tie_weights=bool(config.tie_word_embeddings),
            dropout=float(getattr(config, "attention_dropout", 0.0)),
            rms_norm_eps=float(config.rms_norm_eps),
        )


def _olm_prefix(layer: int) -> str:
    return f"blocks.1.stack.{layer}"


def _hf_prefix(layer: int) -> str:
    return f"model.layers.{layer}"


def hf_to_olm_state(
    hf_state: Mapping[str, torch.Tensor], num_layers: int
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {
        "blocks.0.embedding.weight": hf_state["model.embed_tokens.weight"],
        "blocks.2.weight": hf_state["model.norm.weight"],
    }
    for layer in range(num_layers):
        olm = _olm_prefix(layer)
        hf = _hf_prefix(layer)
        attention = f"{olm}.blocks.0.block.blocks.1"
        feedforward = f"{olm}.blocks.1.block.blocks.1"
        mapped[f"{olm}.blocks.0.block.blocks.0.weight"] = hf_state[
            f"{hf}.input_layernorm.weight"
        ]
        for projection in ("q_proj", "k_proj", "v_proj"):
            mapped[f"{attention}.{projection}.weight"] = hf_state[
                f"{hf}.self_attn.{projection}.weight"
            ]
            mapped[f"{attention}.{projection}.bias"] = hf_state[
                f"{hf}.self_attn.{projection}.bias"
            ]
        mapped[f"{attention}.out_proj.weight"] = hf_state[
            f"{hf}.self_attn.o_proj.weight"
        ]
        mapped[f"{olm}.blocks.1.block.blocks.0.weight"] = hf_state[
            f"{hf}.post_attention_layernorm.weight"
        ]
        # OLM SwiGLU chunks [value, gate]; HF computes SiLU(gate) * up.
        mapped[f"{feedforward}.up_proj.weight"] = torch.cat(
            [
                hf_state[f"{hf}.mlp.up_proj.weight"],
                hf_state[f"{hf}.mlp.gate_proj.weight"],
            ],
            dim=0,
        )
        mapped[f"{feedforward}.down_proj.weight"] = hf_state[
            f"{hf}.mlp.down_proj.weight"
        ]
    if "lm_head.weight" in hf_state:
        mapped["blocks.3.blocks.1.weight"] = hf_state["lm_head.weight"]
    else:
        mapped["blocks.3.blocks.1.weight"] = hf_state["model.embed_tokens.weight"]
    return mapped


def _strip_wrapping_prefixes(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = dict(state)
    for prefix in ("module.", "_orig_mod."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def olm_to_hf_state(
    olm_state: Mapping[str, torch.Tensor], num_layers: int
) -> dict[str, torch.Tensor]:
    source = _strip_wrapping_prefixes(olm_state)
    mapped: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": source["blocks.0.embedding.weight"],
        "model.norm.weight": source["blocks.2.weight"],
        "lm_head.weight": source["blocks.3.blocks.1.weight"],
    }
    for layer in range(num_layers):
        olm = _olm_prefix(layer)
        hf = _hf_prefix(layer)
        attention = f"{olm}.blocks.0.block.blocks.1"
        feedforward = f"{olm}.blocks.1.block.blocks.1"
        mapped[f"{hf}.input_layernorm.weight"] = source[
            f"{olm}.blocks.0.block.blocks.0.weight"
        ]
        for projection in ("q_proj", "k_proj", "v_proj"):
            mapped[f"{hf}.self_attn.{projection}.weight"] = source[
                f"{attention}.{projection}.weight"
            ]
            mapped[f"{hf}.self_attn.{projection}.bias"] = source[
                f"{attention}.{projection}.bias"
            ]
        mapped[f"{hf}.self_attn.o_proj.weight"] = source[
            f"{attention}.out_proj.weight"
        ]
        mapped[f"{hf}.post_attention_layernorm.weight"] = source[
            f"{olm}.blocks.1.block.blocks.0.weight"
        ]
        combined = source[f"{feedforward}.up_proj.weight"]
        value, gate = combined.chunk(2, dim=0)
        mapped[f"{hf}.mlp.up_proj.weight"] = value
        mapped[f"{hf}.mlp.gate_proj.weight"] = gate
        mapped[f"{hf}.mlp.down_proj.weight"] = source[
            f"{feedforward}.down_proj.weight"
        ]
    return mapped


def load_pretrained_olm_qwen(
    model_name: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[Qwen2Model, Any]:
    config = AutoConfig.from_pretrained(model_name)
    model = build_olm_qwen(config, dtype=dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    mapped = hf_to_olm_state(hf_model.state_dict(), int(config.num_hidden_layers))
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incomplete HF-to-OLM conversion: {incompatible}")
    del mapped, hf_model
    gc.collect()
    return model, config


def export_olm_to_hf(
    olm_state: Mapping[str, torch.Tensor],
    config: Any,
    tokenizer_source: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with default_dtype(torch.bfloat16):
        hf_model = AutoModelForCausalLM.from_config(config)
    mapped = olm_to_hf_state(olm_state, int(config.num_hidden_layers))
    incompatible = hf_model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incomplete OLM-to-HF conversion: {incompatible}")
    hf_model.tie_weights()
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)


def model_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    unwrapped = model
    if hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod
    return {key: value.detach().cpu() for key, value in unwrapped.state_dict().items()}
