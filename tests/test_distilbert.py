"""(j) DistilBERT: torch-free helpers always; the training loop on a tiny random model only if torch and
transformers are installed (no GPU, no download, no trained checkpoint needed)."""
import pytest

from src import distilbert as db
from src.config import load_config


def test_config_sized_for_4gb_gpu():
    c = load_config()["distilbert"]
    assert c["max_len"] <= 64 and c["batch_size"] <= 32 and c["base_model"] == "distilbert-base-uncased"
    assert c["out_dir"].startswith("models/")  # models/* is gitignored


def test_label_maps_roundtrip():
    l2i, i2l = db.label_maps(["a", "b", "c"])
    assert l2i == {"a": 0, "b": 1, "c": 2} and all(i2l[l2i[x]] == x for x in "abc")


def test_linear_warmup_decay():
    f = db.linear_warmup_decay(10, 110)
    assert f(0) == 0 and f(5) == pytest.approx(0.5) and f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5) and f(110) == 0 and f(200) == 0
    assert db.linear_warmup_decay(0, 10)(0) == 1.0


def test_early_stopping():
    es = db.EarlyStopping(patience=2)
    assert es.step(1.0, 1) and es.step(0.8, 2)
    assert not es.step(0.85, 3) and not es.should_stop
    assert not es.step(0.81, 4) and es.should_stop
    assert es.best == 0.8 and es.best_epoch == 2


def test_early_stopping_min_delta():
    es = db.EarlyStopping(patience=1, min_delta=0.1)
    es.step(1.0, 1)
    assert not es.step(0.95, 2) and es.should_stop


def test_no_decay_names():
    assert db.no_decay("distilbert.embeddings.LayerNorm.weight") and db.no_decay("classifier.bias")
    assert not db.no_decay("classifier.weight")


def test_missing_checkpoint_message(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    with pytest.raises(FileNotFoundError, match="train_distilbert"):
        db.DistilBertIntentClassifier(tmp_path)


def _tiny_model(n_labels=3):
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    cfg = DistilBertConfig(vocab_size=50, dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=16,
                           num_labels=n_labels)
    return DistilBertForSequenceClassification(cfg)


def test_training_loop_reduces_loss_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    db.set_seed(0)
    model = _tiny_model()
    ids = torch.randint(1, 50, (48, 8))
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    y = (ids[:, 0] % 3).tolist()  # learnable from the first token
    loader = db.make_loader(enc, y, 16, True, 0)
    opt = db.make_optimizer(model, 5e-3, 0.01)
    sched = db.make_scheduler(opt, 2, 60)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    first = db.evaluate(model, db.make_loader(enc, y, 16, False, 0), "cpu", False)[0]
    for _ in range(20):
        db.train_one_epoch(model, loader, opt, sched, scaler, "cpu", False, 1.0)
    last, preds = db.evaluate(model, db.make_loader(enc, y, 16, False, 0), "cpu", False)
    assert last < first and len(preds) == 48


def test_seed_control_is_reproducible():
    torch = pytest.importorskip("torch")
    db.set_seed(123)
    a = torch.rand(3)
    db.set_seed(123)
    assert torch.equal(a, torch.rand(3))
