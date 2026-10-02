"""Banking77: fine-tuning vs prompting. Streamlit app (Classify, Evaluation, Training, Monitoring).

Run:  .venv\\Scripts\\streamlit run streamlit_app.py
No mock fallbacks: LLM systems need Ollama; if it is down or a model is missing, the app says so.
"""
from __future__ import annotations

import json
import os

import altair as alt
import pandas as pd
import psutil
import streamlit as st

from src import llm
from src.config import load_config, path
from src.systems import TFIDF_KEY, classify, system_keys, system_label, system_model
from src.tracing import Tracer

st.set_page_config(page_title="Banking77: fine-tuning vs prompting", layout="wide")
cfg = load_config()
tracer = Tracer(path("traces_db"))

# Fixed colour per system (colour follows the entity, never its rank). Validated default categorical order.
COLORS = {"tfidf_logreg": "#2a78d6", "zero_shot_7b": "#eb6834", "few_shot_7b": "#1baf7a",
          "base_1p5b": "#eda100", "finetuned_1p5b": "#e87ba4", "zero_shot_3b": "#008300"}
ATTRIBUTION = ("Data: Banking77 (Casanueva et al., 2020, PolyAI), CC BY 4.0. "
               "https://github.com/PolyAI-LDN/task-specific-datasets")


@st.cache_data(ttl=30, show_spinner=False)
def availability() -> dict[str, str]:
    """'ok' or a reason, per system. Only asks Ollama for its model list (no generation)."""
    out = {TFIDF_KEY: "ok" if (path("models") / "tfidf_logreg.joblib").exists() else "run scripts/run_baseline.py"}
    try:
        tags = llm.list_models()
    except llm.OllamaUnavailable as e:
        tags, err = None, str(e)
    for key, sc in cfg["llm"]["systems"].items():
        if tags is None:
            out[key] = err
        elif sc["model"] in tags or f"{sc['model']}:latest" in tags:
            out[key] = "ok"
        elif sc["mode"] == "finetuned":
            out[key] = "pending: run the Colab notebook, then scripts/import_finetuned.ps1"
        else:
            out[key] = f"model not pulled: ollama pull {sc['model']}"
    return out


PAGES = ["Classify", "Evaluation", "Training", "Monitoring"]
_qp = st.query_params.get("page", "Classify")  # ?page=Evaluation deep link (used for screenshots)
page = st.sidebar.radio("Page", PAGES, index=PAGES.index(_qp) if _qp in PAGES else 0)
st.sidebar.caption(f"App process RAM: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.0f} MB")
st.sidebar.caption(ATTRIBUTION)

if page == "Classify":
    st.title("Classify a banking support message")
    st.write("Each available system classifies the message into one of 77 intents; latency is measured per request.")
    av = availability()
    text = st.text_area("Message", "I still haven't received my new card, when will it arrive?", height=80)
    chosen = [k for k in system_keys()
              if st.checkbox(system_label(k), value=(av[k] == "ok"), disabled=(av[k] != "ok"),
                             help=None if av[k] == "ok" else av[k], key=f"cb_{k}")]
    for k in system_keys():
        if av[k] != "ok":
            st.caption(f"{system_label(k)}: {av[k]}")
    if st.button("Classify", type="primary") and text.strip():
        rows = []
        for k in chosen:
            try:
                r = classify(text.strip(), k)
                err = None
            except Exception as e:  # noqa: BLE001  (show the real error; no fallback)
                r, err = {}, f"{type(e).__name__}: {e}"
            tracer.log(source="app", system=k, model=system_model(k), input=text.strip(),
                       prediction=r.get("prediction"), valid=int(bool(r.get("valid"))), raw_output=r.get("raw"),
                       t_retrieve=r.get("t_retrieve"), t_model=r.get("t_model"), t_total=r.get("t_total"),
                       prompt_tokens=r.get("prompt_tokens"), completion_tokens=r.get("completion_tokens"), error=err)
            rows.append({"system": system_label(k), "prediction": r.get("prediction") or "-",
                         "valid label": "yes" if r.get("valid") else "no",
                         "latency (ms)": round(r["t_total"] * 1000, 1) if r.get("t_total") is not None else None,
                         "error": err or ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

elif page == "Evaluation":
    st.title("Evaluation")
    comp_p = path("results") / "comparison.csv"
    if not comp_p.exists():
        st.error("No results yet. Run: .venv\\Scripts\\python scripts\\summarize_results.py")
        st.stop()
    comp = pd.read_csv(comp_p)
    split = st.radio("Split", ["test_subset", "test"], horizontal=True,
                     format_func=lambda s: {"test_subset": "Stratified test subset (770 = 10 per class)",
                                            "test": "Full test set (3,080)"}[s])
    st.caption("All systems are scored on both the stratified subset and the full test set. "
               "Cost per 1,000 requests = measured mean seconds per request x an ASSUMED cloud price "
               f"(${cfg['cost']['cpu_instance_usd_per_hour']}/h CPU instance, config.yaml). "
               "LLM latencies: GPU laptop (RTX 3050, 4 GB VRAM; the 7B is split GPU/CPU). "
               "TF-IDF latency: build laptop CPU.")
    d = comp[comp["split"] == split].copy()
    show = ["label", "status", "n", "accuracy", "macro_f1", "invalid_rate", "p50_ms", "p95_ms", "throughput_rps",
            "cost_per_1k_usd", "size_on_disk_mb", "mem_loaded_mb"]
    st.dataframe(d[[c for c in show if c in d]].rename(columns={"label": "system"}), hide_index=True,
                 use_container_width=True,
                 column_config={"accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "macro_f1": st.column_config.NumberColumn(format="%.3f"),
                                "invalid_rate": st.column_config.NumberColumn(format="%.3f"),
                                "p50_ms": st.column_config.NumberColumn(format="%.1f"),
                                "p95_ms": st.column_config.NumberColumn(format="%.1f"),
                                "throughput_rps": st.column_config.NumberColumn(format="%.1f"),
                                "cost_per_1k_usd": st.column_config.NumberColumn(format="$%.5f"),
                                "size_on_disk_mb": st.column_config.NumberColumn(format="%.0f"),
                                "mem_loaded_mb": st.column_config.NumberColumn(format="%.0f")})
    done = d[d["status"] == "done"]
    st.subheader("Accuracy vs latency (p50, log scale)")
    if done.empty:
        st.info("No systems scored on this split yet.")
    else:
        keys = list(COLORS)
        chart = alt.Chart(done).mark_circle(size=140, opacity=1, stroke="white", strokeWidth=2).encode(
            x=alt.X("p50_ms:Q", scale=alt.Scale(type="log"), title="p50 latency per request (ms, log)"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False), title="accuracy", axis=alt.Axis(format=".2f")),
            color=alt.Color("system:N", scale=alt.Scale(domain=keys, range=[COLORS[k] for k in keys]),
                            legend=alt.Legend(title="system", orient="bottom")),
            tooltip=["label", alt.Tooltip("accuracy:Q", format=".3f"), alt.Tooltip("macro_f1:Q", format=".3f"),
                     alt.Tooltip("p50_ms:Q", format=".1f"), alt.Tooltip("p95_ms:Q", format=".1f")])
        st.altair_chart(chart.properties(height=320).configure_axis(grid=False), use_container_width=True)
    pcf_p = path("results") / "per_class_f1.csv"
    if pcf_p.exists() and not done.empty:
        pcf = pd.read_csv(pcf_p).set_index("label")
        cols = [c for c in pcf.columns if c.endswith(f"|{split}")]
        sys_pick = st.selectbox("Per-class F1 for", [c.split("|")[0] for c in cols], format_func=system_label)
        s = pcf[f"{sys_pick}|{split}"].sort_values()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lowest per-class F1 (15)")
            low = s.head(15).reset_index().rename(columns={f"{sys_pick}|{split}": "f1"})
            st.altair_chart(alt.Chart(low).mark_bar(color=COLORS.get(sys_pick, "#2a78d6"),
                                                    cornerRadiusEnd=4).encode(
                x=alt.X("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=320)), tooltip=["label", alt.Tooltip("f1:Q", format=".3f")])
                .properties(height=380).configure_axis(grid=False), use_container_width=True)
        with c2:
            st.subheader("Most-confused intents")
            conf = json.loads((path("results") / "confusions.json").read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame(conf.get(f"{sys_pick}|{split}", [])), hide_index=True,
                         use_container_width=True)

elif page == "Training":
    st.title("LoRA training (Colab)")
    log_p = path("models") / "training_log.json"
    ft = cfg["finetune"]
    if not log_p.exists():
        st.warning("(e) pending: run the Colab notebook. Steps: open notebooks/finetune_lora_colab.ipynb in Colab, "
                   "Runtime -> T4 GPU, Run all, download banking77_outputs.zip, unzip into models/, then run "
                   "scripts/import_finetuned.ps1 and scripts/eval_finetuned.py.")
        st.subheader("Planned settings (config.yaml)")
        st.json(ft)
    else:
        log = json.loads(log_p.read_text(encoding="utf-8"))
        c = st.columns(4)
        c[0].metric("Validation accuracy", f"{log['val_accuracy']:.3f}")
        c[1].metric("GPU", log["gpu"])
        c[2].metric("Training time", f"{log['train_seconds'] / 60:.1f} min")
        c[3].metric("Epochs completed", f"{log['epochs_completed']:.2f}")
        tl = pd.DataFrame(log["train_loss"]).assign(series="train")
        vl = pd.DataFrame(log["val_loss"]).assign(series="validation")
        both = pd.concat([tl, vl])
        st.altair_chart(alt.Chart(both).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
            x=alt.X("step:Q", title="step"), y=alt.Y("loss:Q", title="loss"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["train", "validation"],
                                                        range=["#2a78d6", "#eb6834"])),
            tooltip=["series", "step", alt.Tooltip("loss:Q", format=".4f")]).properties(height=340)
            .configure_axis(grid=False), use_container_width=True)
        gpu_h = log.get("gpu_hours", 0)
        st.caption(f"Training cost: {gpu_h:.2f} GPU-hours on {log['gpu']} (free Colab). At the ASSUMED "
                   f"\\${cfg['cost']['gpu_instance_usd_per_hour']}/h T4 price that is "   # \$: plain dollar, not LaTeX
                   f"\\${gpu_h * cfg['cost']['gpu_instance_usd_per_hour']:.2f}.")
        st.json(log["settings"])

else:
    st.title("Monitoring")
    df = tracer.read(500)
    if df.empty:
        st.info("No requests traced yet. Classify a message on the Classify page.")
    else:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        agg = df.groupby("system").agg(requests=("id", "count"), p50_ms=("t_total", lambda s: s.median() * 1000),
                                       errors=("error", lambda s: s.notna().sum()),
                                       invalid=("valid", lambda s: (s == 0).sum())).reset_index()
        st.dataframe(agg, hide_index=True, use_container_width=True)
        st.dataframe(df[["time", "source", "system", "model", "input", "prediction", "valid", "t_retrieve",
                         "t_model", "t_total", "prompt_tokens", "completion_tokens", "error"]],
                     hide_index=True, use_container_width=True)
