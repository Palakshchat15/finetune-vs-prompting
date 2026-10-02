"""Prompt templates. The SAME functions are used by local inference, the eval scripts and the
Colab notebook (the notebook carries a verbatim copy; tests/test_prompts.py checks they match).

Design:
- Prompted systems (b, c, d) get a FIXED system prompt listing all 77 labels, so Ollama can reuse
  the cached prompt prefix across requests. Few-shot examples go in the user turn, after the fixed
  prefix, so the prefix stays cacheable.
- They answer through Ollama structured output: a JSON schema whose `intent` is an enum of the labels.
- The fine-tuned system (e) uses a SHORT system prompt without the label list (the labels are
  learned in training; this keeps training sequences under 256 tokens and inference fast). Its target
  is the exact label string, generated as plain text (unconstrained), so invalid outputs are measurable.
"""
from __future__ import annotations

import json

FINETUNE_SYSTEM = ("You classify customer-support messages for a bank into one of 77 intents. "
                   "Reply with the intent label only.")


def label_list_system(labels: list[str]) -> str:
    return ("You classify customer-support messages for a bank into exactly one intent.\n"
            "Choose the single best intent from this list (use the label exactly as written):\n"
            + "\n".join(labels)
            + '\n\nReply in JSON: {"intent": "<label>"}')


def intent_schema(labels: list[str]) -> dict:
    """JSON schema for Ollama's `format` field (structured output restricted to the labels)."""
    return {"type": "object", "properties": {"intent": {"type": "string", "enum": list(labels)}},
            "required": ["intent"]}


def zero_shot_messages(text: str, labels: list[str]) -> list[dict]:
    return [{"role": "system", "content": label_list_system(labels)},
            {"role": "user", "content": f"Message: {text}"}]


def few_shot_messages(text: str, labels: list[str], examples: list[tuple[str, str]]) -> list[dict]:
    shots = "\n".join(f"Message: {t}\nIntent: {lab}" for t, lab in examples)
    user = f"Labelled examples of similar messages:\n{shots}\n\nNow classify this message.\nMessage: {text}"
    return [{"role": "system", "content": label_list_system(labels)}, {"role": "user", "content": user}]


def finetune_messages(text: str, label: str | None = None) -> list[dict]:
    """Chat messages for training (label given) and for inference (label None)."""
    msgs = [{"role": "system", "content": FINETUNE_SYSTEM}, {"role": "user", "content": text}]
    if label is not None:
        msgs.append({"role": "assistant", "content": label})
    return msgs


# Qwen2.5 ChatML, written out so the Ollama Modelfile matches the training template exactly.
# System messages are rendered once via .System (skipped inside the loop to avoid duplicates).
OLLAMA_TEMPLATE = ("{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
                   "{{- range .Messages }}{{- if ne .Role \"system\" }}"
                   "<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}{{- end }}"
                   "<|im_start|>assistant\n")


def render_chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Python rendering of the ChatML prompt, identical to Qwen2.5's HF chat template when a system
    message is present, and to OLLAMA_TEMPLATE. Used by tests and for documentation."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def parse_structured(content: str, labels: set[str]) -> tuple[str | None, bool]:
    """Return (label, valid). Invalid = not JSON, missing key, or not one of the labels."""
    try:
        lab = json.loads(content).get("intent")
    except (json.JSONDecodeError, AttributeError):
        return None, False
    if isinstance(lab, str) and lab.strip() in labels:
        return lab.strip(), True
    return (lab if isinstance(lab, str) else None), False


def parse_plain(content: str, labels: set[str]) -> tuple[str, bool]:
    """Fine-tuned output: exact label string expected; only whitespace/quotes/trailing period stripped."""
    lab = content.strip()
    lab = lab.splitlines()[0].strip() if lab else lab
    lab = lab.rstrip(".").strip().strip('"\'`').strip()
    return lab, lab in labels
