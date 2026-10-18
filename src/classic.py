"""(f)-(i) Classic NLP baselines: TF-IDF + multinomial Naive Bayes, TF-IDF + linear SVM, character n-gram
TF-IDF + linear SVM, and spaCy lemmatisation (+ optional stop-word removal) + linear SVM.

Same protocol as (a) (src/baseline.py): hyperparameters chosen on validation only, refit on train, test
scored once. spaCy is imported lazily, so the other three systems (and the tests) need only scikit-learn.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import load_config
from .metrics import summarize

# --------------------------------------------------------------------------------------------------
# spaCy preprocessing
# --------------------------------------------------------------------------------------------------
_NLP = {}            # model name -> loaded spaCy pipeline (not pickled with the sklearn model)
_MEMO: dict = {}     # (model name, text) -> [(lemma, is_stop), ...]; shared by every grid fit


def load_spacy(model: str):
    """Load a spaCy pipeline once, with the parser and NER disabled (only the lemmatizer is needed)."""
    if model not in _NLP:
        try:
            import spacy  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("spaCy is not installed. Run: .venv\\Scripts\\pip install -r requirements-nlp.txt") from e
        try:
            _NLP[model] = spacy.load(model, disable=["parser", "ner"])
        except OSError as e:
            raise OSError(f"spaCy model '{model}' is missing. Run: .venv\\Scripts\\pip install -r "
                          "requirements-nlp.txt (it installs en_core_web_sm from its pinned wheel URL)") from e
    return _NLP[model]


def clear_memo() -> None:
    """Forget memoised lemmas (used before latency timing, so spaCy really runs per message)."""
    _MEMO.clear()


def _cache_path(model: str) -> Path:
    from .config import path  # noqa: PLC0415  (creates data/cache only when actually used)
    return path("cache") / f"spacy_lemmas__{model}.jsonl"


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_disk_cache(model: str) -> None:
    p = _cache_path(model)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line (killed mid-write)
            _MEMO[(model, rec["text"])] = [tuple(t) for t in rec["tokens"]]


def analyse_texts(texts: list[str], model: str, nlp=None, disk_cache: bool = False) -> list[list[tuple[str, bool]]]:
    """[(lowercase lemma, is_stop), ...] per text; punctuation and whitespace tokens dropped.
    Results are memoised per text (and optionally in data/cache/spacy_lemmas__<model>.jsonl)."""
    if disk_cache and not _MEMO:
        _load_disk_cache(model)
    todo = list(dict.fromkeys(t for t in texts if (model, t) not in _MEMO))
    if todo:
        nlp = nlp or load_spacy(model)
        new = []
        for t, doc in zip(todo, nlp.pipe(todo, batch_size=256)):
            toks = [(tok.lemma_.lower().strip(), bool(tok.is_stop)) for tok in doc
                    if not (tok.is_punct or tok.is_space) and tok.lemma_.strip()]
            _MEMO[(model, t)] = toks
            new.append({"key": _text_key(t), "text": t, "tokens": toks})
        if disk_cache:
            with open(_cache_path(model), "a", encoding="utf-8") as f:
                for rec in new:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return [_MEMO[(model, t)] for t in texts]


def lemma_string(tokens: list[tuple[str, bool]], remove_stop: bool, keep: set[str]) -> str:
    return " ".join(lem for lem, stop in tokens if not (remove_stop and stop and lem not in keep))


class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    """sklearn transformer: raw text -> space-joined lemmas (stop words optionally removed).
    Only its parameters are pickled; the spaCy pipeline is reloaded lazily after joblib.load."""

    def __init__(self, model: str = "en_core_web_sm", remove_stop: bool = True, keep_words: tuple = (),
                 disk_cache: bool = False):
        self.model = model
        self.remove_stop = remove_stop
        self.keep_words = keep_words
        self.disk_cache = disk_cache

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        keep = set(self.keep_words)
        toks = analyse_texts([str(x) for x in X], self.model, disk_cache=self.disk_cache)
        return [lemma_string(t, self.remove_stop, keep) for t in toks]


# --------------------------------------------------------------------------------------------------
# Pipelines and validation-only model selection
# --------------------------------------------------------------------------------------------------
def system_keys() -> list[str]:
    return list(load_config()["classic"]["systems"])


def param_grid(key: str) -> list[dict]:
    c = load_config()["classic"]
    sc = c["systems"][key]
    vals = c["nb_alpha_grid"] if sc["clf"] == "nb" else c["svm_C_grid"]
    name = "alpha" if sc["clf"] == "nb" else "C"
    if sc["preprocess"] == "spacy":
        return [{name: v, "remove_stop": rs} for rs in c["spacy_remove_stop_grid"] for v in vals]
    return [{name: v} for v in vals]


def make_pipeline(key: str, params: dict, disk_cache: bool = False) -> Pipeline:
    c = load_config()["classic"]
    sc = c["systems"][key]
    steps = []
    if sc["preprocess"] == "spacy":
        steps.append(("spacy", SpacyLemmatizer(model=c["spacy_model"], remove_stop=params["remove_stop"],
                                               keep_words=tuple(c["spacy_keep_words"]), disk_cache=disk_cache)))
    if sc["features"] == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c["char_ngram_range"]),
                              sublinear_tf=c["sublinear_tf"], min_df=c["min_df"], lowercase=True)
    else:
        vec = TfidfVectorizer(ngram_range=tuple(c["word_ngram_range"]), sublinear_tf=c["sublinear_tf"],
                              min_df=c["min_df"], lowercase=True)
    steps.append(("tfidf", vec))
    if sc["clf"] == "nb":
        steps.append(("clf", MultinomialNB(alpha=params["alpha"])))
    elif sc["clf"] == "svm":
        steps.append(("clf", LinearSVC(C=params["C"], max_iter=20000, random_state=load_config()["data"]["seed"])))
    else:
        raise ValueError(f"unknown classifier {sc['clf']}")
    return Pipeline(steps)


def select_on_val(make, grid: list[dict], train_x, train_y, val_x, val_y, labels: list[str],
                  log=print) -> tuple[dict, list[dict]]:
    """Fit make(params) on train for every params in grid, score on validation, return (best, all rows).
    Best = highest val macro-F1, ties broken by accuracy, then by grid order (first wins). Test is never seen."""
    rows = []
    for params in grid:
        m = make(params).fit(train_x, train_y)
        s = summarize(list(val_y), list(m.predict(val_x)), labels)
        rows.append({**params, **s})
        log(f"  {params} val acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")
    best_i = max(range(len(rows)), key=lambda i: (rows[i]["macro_f1"], rows[i]["accuracy"], -i))
    best = {k: rows[best_i][k] for k in grid[best_i]}
    return best, rows


def grid_edge_note(best: dict, grid: list[dict]) -> str | None:
    """Warn when the chosen value sits at the edge of its numeric grid (as was done for (a))."""
    for name in ("C", "alpha"):
        if name in best:
            vals = sorted({g[name] for g in grid})
            if best[name] in (vals[0], vals[-1]):
                return f"{name}={best[name]} is at the edge of the grid {vals}; consider widening it (on val only)."
    return None
