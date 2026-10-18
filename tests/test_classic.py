"""(f)-(i) classic baselines: pipelines, validation-only selection, spaCy preprocessing (with a fake spaCy)."""
from types import SimpleNamespace

import pytest

from src import classic
from src.config import load_config

TOY_X = ["my card has not arrived", "card still not here", "where is my new card", "card delivery when",
         "top up failed", "my top up did not work", "top-up was declined", "why did my top up fail"] * 3
TOY_Y = (["card_arrival"] * 4 + ["top_up_failed"] * 4) * 3
LABELS = ["card_arrival", "top_up_failed"]


def test_config_registers_four_systems():
    assert classic.system_keys() == ["nb_tfidf", "svm_tfidf", "svm_char", "spacy_lemma_svm"]


@pytest.mark.parametrize("key", ["nb_tfidf", "svm_tfidf", "svm_char"])
def test_pipelines_fit_and_predict_known_labels(key):
    params = classic.param_grid(key)[0]
    m = classic.make_pipeline(key, params).fit(TOY_X, TOY_Y)
    assert set(m.predict(["card not arrived", "top up failed again"])) <= set(LABELS)


def test_char_system_uses_char_ngrams():
    m = classic.make_pipeline("svm_char", {"C": 1})
    assert m.named_steps["tfidf"].analyzer == "char_wb"
    assert tuple(m.named_steps["tfidf"].ngram_range) == tuple(load_config()["classic"]["char_ngram_range"])


def test_param_grids():
    c = load_config()["classic"]
    assert [g["alpha"] for g in classic.param_grid("nb_tfidf")] == c["nb_alpha_grid"]
    assert [g["C"] for g in classic.param_grid("svm_tfidf")] == c["svm_C_grid"]
    g = classic.param_grid("spacy_lemma_svm")
    assert len(g) == len(c["svm_C_grid"]) * len(c["spacy_remove_stop_grid"])
    assert {x["remove_stop"] for x in g} == set(c["spacy_remove_stop_grid"])


def test_select_on_val_uses_only_val_and_picks_best():
    calls = []

    class Fake:
        def __init__(self, p):
            self.p = p

        def fit(self, x, y):
            calls.append(("fit", list(x)))
            return self

        def predict(self, x):
            calls.append(("predict", list(x)))
            return ["a"] * len(x) if self.p["C"] == 3 else ["b"] * len(x)

    best, rows = classic.select_on_val(Fake, [{"C": 1}, {"C": 3}, {"C": 10}], ["tr"], ["a"], ["v1", "v2"],
                                       ["a", "a"], ["a", "b"], log=lambda *_: None)
    assert best == {"C": 3} and len(rows) == 3
    assert all(x in (["tr"], ["v1", "v2"]) for _, x in calls)  # only train is fitted, only val is scored


def test_select_on_val_tie_keeps_first():
    class Same:
        def __init__(self, p):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return ["a"] * len(x)

    best, _ = classic.select_on_val(Same, [{"C": 1}, {"C": 3}], [], [], ["v"], ["a"], ["a"], log=lambda *_: None)
    assert best == {"C": 1}


def test_grid_edge_note():
    grid = [{"C": 1}, {"C": 3}, {"C": 10}]
    assert classic.grid_edge_note({"C": 10}, grid) and classic.grid_edge_note({"C": 3}, grid) is None


class FakeNLP:
    """Minimal stand-in for a spaCy pipeline: lemma = lowercase word without trailing 's'."""
    STOP = {"my", "the", "is", "not", "why", "has"}

    def pipe(self, texts, batch_size=256):
        for t in texts:
            yield [SimpleNamespace(lemma_=w.lower().rstrip("s") if len(w) > 3 else w.lower(),
                                   is_stop=w.lower() in self.STOP, is_punct=w in {"?", "!", "."}, is_space=False)
                   for w in t.replace("?", " ?").split()]


@pytest.fixture
def fake_spacy(monkeypatch):
    classic.clear_memo()
    monkeypatch.setattr(classic, "load_spacy", lambda model: FakeNLP())
    yield
    classic.clear_memo()


def test_spacy_lemmatizer_removes_stop_words_but_keeps_intent_words(fake_spacy):
    t = classic.SpacyLemmatizer(remove_stop=True, keep_words=("not", "why"))
    assert t.transform(["Why has my card not arrived?"]) == ["why card not arrived"]
    t2 = classic.SpacyLemmatizer(remove_stop=False)
    assert t2.transform(["Why has my card not arrived?"]) == ["why has my card not arrived"]


def test_spacy_pipeline_end_to_end_with_fake_spacy(fake_spacy):
    m = classic.make_pipeline("spacy_lemma_svm", {"C": 1, "remove_stop": True}).fit(TOY_X, TOY_Y)
    assert m.predict(["my cards have not arrived"])[0] in LABELS
    assert m.named_steps["spacy"].keep_words == tuple(load_config()["classic"]["spacy_keep_words"])


def test_spacy_memo_and_clear(fake_spacy):
    classic.analyse_texts(["cards"], "m")
    assert ("m", "cards") in classic._MEMO
    classic.clear_memo()
    assert not classic._MEMO


def test_registry_includes_new_systems():
    from src.systems import system_family, system_keys, system_label, system_model
    keys = system_keys()
    assert keys[:5] == ["tfidf_logreg", "zero_shot_7b", "few_shot_7b", "base_1p5b", "zero_shot_3b"]
    for k in ("nb_tfidf", "svm_tfidf", "svm_char", "spacy_lemma_svm", "distilbert_pt"):
        assert k in keys and system_label(k)
    assert system_family("svm_char") == "classic" and system_model("svm_char") == "sklearn"
    assert system_family("distilbert_pt") == "distilbert" and system_model("distilbert_pt") == "distilbert-base-uncased"
    assert system_family("few_shot_7b") == "llm"


def test_summarize_adds_not_run_rows_and_classic_rows(tmp_path, monkeypatch):
    """New systems only add rows; missing results become 'not run yet' / 'pending' rows."""
    import importlib.util
    import json

    from src.config import ROOT, path
    if not (path("splits") / "labels.json").exists():
        pytest.skip("run scripts/prepare_data.py")
    spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "path", lambda key: tmp_path)
    monkeypatch.setattr(m, "cache_file", lambda s, mo, sp: tmp_path / f"{s}_{sp}.jsonl")
    res = {"model_size_mb": 1.0}
    for split in m.SPLITS:
        res[split] = {"n": 10, "accuracy": 0.5, "macro_f1": 0.4, "p50_ms": 1.0, "p95_ms": 2.0, "throughput_rps": 9.0,
                      "mean_ms": 1.1, "cost_per_1k_usd_cpu_assumption": 0.0, "per_class_f1": {"a": 0.4},
                      "top_confusions": []}
    (tmp_path / "classic_svm_tfidf.json").write_text(json.dumps(res), encoding="utf-8")
    rows, pcf, conf = [], {}, {}
    m.add_new_systems(load_config(), ["a"], rows, pcf, conf)
    by = {(r["system"], r["split"]): r for r in rows}
    assert by[("svm_tfidf", "test")]["status"] == "done" and by[("svm_tfidf", "test")]["accuracy"] == 0.5
    assert by[("nb_tfidf", "test")]["status"].startswith("not run yet")
    assert by[("distilbert_pt", "test")]["status"].startswith("pending")
    assert "svm_tfidf|test" in pcf and "nb_tfidf|test" not in pcf
