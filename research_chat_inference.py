"""
Lazy-loaded Hugging Face causal LM for research chat (CPU-friendly small instruct model).

Install: pip install -r requirements-research-chat.txt

Env:
  RESEARCH_CHAT_HF_MODEL — default HuggingFaceTB/SmolLM2-360M-Instruct
  RESEARCH_CHAT_DEVICE   — cpu | cuda:0 (default cpu)
"""

from __future__ import annotations

import logging
import os
from threading import Lock

logger = logging.getLogger(__name__)

_lock = Lock()
_bundle: tuple[object, object] | None = None

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _load_bundle() -> tuple[object, object]:
    global _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Research chat needs torch + transformers. "
                "Install: pip install -r requirements-research-chat.txt"
            ) from exc

        model_id = os.getenv("RESEARCH_CHAT_HF_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        device = os.getenv("RESEARCH_CHAT_DEVICE", "cpu").strip() or "cpu"

        logger.info("Loading research chat model %s on %s (first request may take a while)", model_id, device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        _bundle = (tokenizer, model)
        return _bundle


def generate_reply(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """messages: OpenAI-style roles system|user|assistant (model must support chat template)."""
    import torch

    tokenizer, model = _load_bundle()
    device = next(model.parameters()).device

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        # Fallback for tokenizers without chat_template
        flat = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        flat += "\n\nASSISTANT:"
        input_ids = tokenizer(flat, return_tensors="pt").input_ids

    input_ids = input_ids.to(device)
    input_len = int(input_ids.shape[-1])

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def is_inference_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
