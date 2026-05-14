"""Two-pass research chat: draft from web context, then final + self-analysis with delimiters."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_chat_inference import generate_reply, is_inference_available

logger = logging.getLogger(__name__)


class ResearchChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=24_000)


class ResearchChatRequest(BaseModel):
    messages: list[ResearchChatMessage] = Field(..., min_length=1, max_length=24)
    web_snippets: str = Field(default="", max_length=48_000)
    last_user_message: str = Field(default="", max_length=24_000)


def _max_tokens(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        n = int(raw)
        return max(32, min(n, 2048))
    except ValueError:
        return default


def _build_history_summary(messages: list[ResearchChatMessage]) -> str:
    tail = messages[-8:]
    lines: list[str] = []
    for m in tail:
        label = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else "System"
        c = m.content.strip()
        if len(c) > 2000:
            c = c[:2000] + "…"
        lines.append(f"{label}: {c}")
    return "\n\n".join(lines) if lines else "(no prior turns)"


def _parse_final_self(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if "###FINAL" in text and "###SELF" in text:
        try:
            after_final = text.split("###FINAL", 1)[1]
            parts = after_final.split("###SELF", 1)
            answer = parts[0].strip()
            self_part = parts[1].strip() if len(parts) > 1 else ""
            if answer:
                return answer, self_part or "_(Empty self-analysis section.)_"
        except Exception:
            logger.warning("Failed to parse ###FINAL / ###SELF blocks")
    # Heuristic: treat whole output as answer if model ignored template
    return text, "_(The model did not emit ###FINAL / ###SELF blocks; showing full output as the answer.)_"


def run_research_chat(req: ResearchChatRequest) -> dict[str, Any]:
    if not is_inference_available():
        raise RuntimeError(
            "Research chat dependencies missing. On the ML host run: pip install -r requirements-research-chat.txt"
        )

    snippets = (req.web_snippets or "").strip() or "(No web search results were supplied.)"
    history = _build_history_summary(req.messages)
    last_q = (req.last_user_message or "").strip()
    if not last_q:
        last_q = req.messages[-1].content.strip()

    draft_tokens = _max_tokens("RESEARCH_CHAT_DRAFT_MAX_NEW_TOKENS", 384)
    refine_tokens = _max_tokens("RESEARCH_CHAT_REFINE_MAX_NEW_TOKENS", 512)

    draft_system = (
        "You are a careful research assistant. You receive public web snippets (may be empty). "
        "Ground factual claims in the snippets when possible; cite bracketed source numbers like [1] if present in the snippet list. "
        "If snippets are missing or irrelevant, say so. Do not invent URLs or quotes."
    )
    draft_user = (
        f"## Web snippets\n{snippets}\n\n## Conversation\n{history}\n\n"
        "Write a clear draft answer addressing only the latest user message."
    )

    draft = generate_reply(
        [{"role": "system", "content": draft_system}, {"role": "user", "content": draft_user}],
        max_new_tokens=draft_tokens,
        temperature=0.55,
        top_p=0.9,
    )

    refine_system = (
        "You improve research answers. Read the web snippets and the draft. "
        "You MUST respond using exactly this structure (including the marker lines):\n"
        "###FINAL\n"
        "(markdown for the user — improved answer)\n"
        "###SELF\n"
        "(markdown: limitations, what snippets did/did not support, confidence, what to verify next)\n"
        "Do not add text before ###FINAL."
    )
    refine_user = (
        f"## Web snippets\n{snippets}\n\n## Latest user question\n{last_q}\n\n## Draft\n{draft}\n\n"
        "Produce ###FINAL then ###SELF as instructed."
    )

    refined = generate_reply(
        [{"role": "system", "content": refine_system}, {"role": "user", "content": refine_user}],
        max_new_tokens=refine_tokens,
        temperature=0.35,
        top_p=0.88,
    )

    answer, self_analysis = _parse_final_self(refined)

    return {
        "answer": answer,
        "self_analysis": self_analysis,
        "draft": draft,
        "model_id": os.getenv("RESEARCH_CHAT_HF_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    }
