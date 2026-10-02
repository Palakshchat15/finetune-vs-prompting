import json

from src import prompts

LABELS = ["card_arrival", "card_linking", "exchange_rate"]


def test_zero_shot_prefix_is_fixed():
    a = prompts.zero_shot_messages("hello", LABELS)
    b = prompts.zero_shot_messages("other message", LABELS)
    assert a[0] == b[0]  # identical system prompt -> Ollama prefix cache reuse
    assert all(l in a[0]["content"] for l in LABELS)


def test_few_shot_keeps_system_fixed_and_includes_examples():
    m = prompts.few_shot_messages("q", LABELS, [("my card?", "card_arrival")])
    assert m[0] == prompts.zero_shot_messages("x", LABELS)[0]
    assert "my card?" in m[1]["content"] and "card_arrival" in m[1]["content"]


def test_schema_enum():
    s = prompts.intent_schema(LABELS)
    assert s["properties"]["intent"]["enum"] == LABELS


def test_parse_structured():
    ls = set(LABELS)
    assert prompts.parse_structured(json.dumps({"intent": "card_linking"}), ls) == ("card_linking", True)
    assert prompts.parse_structured(json.dumps({"intent": "nope"}), ls) == ("nope", False)
    assert prompts.parse_structured("not json", ls) == (None, False)
    assert prompts.parse_structured("[]", ls) == (None, False)


def test_parse_plain():
    ls = set(LABELS)
    assert prompts.parse_plain(" card_arrival\n", ls) == ("card_arrival", True)
    assert prompts.parse_plain('"exchange_rate".', ls) == ("exchange_rate", True)
    assert prompts.parse_plain("Card arrival", ls)[1] is False


def test_finetune_messages_train_vs_infer():
    assert len(prompts.finetune_messages("x")) == 2
    m = prompts.finetune_messages("x", "card_arrival")
    assert m[-1] == {"role": "assistant", "content": "card_arrival"}


def test_chatml_matches_ollama_template_shape():
    s = prompts.render_chatml(prompts.finetune_messages("hi"))
    assert s == (f"<|im_start|>system\n{prompts.FINETUNE_SYSTEM}<|im_end|>\n"
                 "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    t = prompts.OLLAMA_TEMPLATE
    assert t.endswith("<|im_start|>assistant\n") and 'ne .Role "system"' in t
