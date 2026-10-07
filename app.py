"""
app.py
------
Streamlit web application for live Spam Email Detection.

Run with:
    streamlit run app.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.preprocessing import clean_text

# ── Path constants ─────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, "models", "spam_classifier.pkl")
VECTOR_PATH  = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
CM_IMG       = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
COMP_IMG     = os.path.join(OUTPUTS_DIR, "model_comparison.png")
DIST_IMG     = os.path.join(OUTPUTS_DIR, "class_distribution.png")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #f0f2f6; }
[data-testid="stSidebar"]          { background: #1a1f2e; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr       { border-color: #334155; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 50%, #1a4a7a 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero h1  { font-size: 2.4rem; font-weight: 800; margin: 0 0 0.35rem; letter-spacing: -0.5px; }
.hero p   { font-size: 1.05rem; margin: 0; opacity: 0.85; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 0.25rem;
}

/* ── Stat tiles ── */
.stat-tile {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid #e2e8f0;
    height: 100%;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Result boxes ── */
.result-spam {
    background: #fff1f2;
    border: 2px solid #f43f5e;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
}
.result-ham {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
}
.result-emoji  { font-size: 3rem; margin-bottom: 0.4rem; }
.result-title  { font-size: 1.9rem; font-weight: 800; margin: 0.1rem 0; }
.result-sub    { font-size: 1rem; color: #475569; margin-top: 0.3rem; }
.conf-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 0.6rem;
}
.conf-pill-spam { background: #fecdd3; color: #9f1239; }
.conf-pill-ham  { background: #bbf7d0; color: #14532d; }

/* ── Section headings ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
    margin: 1.5rem 0 1rem;
}

/* ── Example cards ── */
.example-card {
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #e2e8f0;
    font-size: 0.92rem;
    color: #334155;
    min-height: 80px;
    line-height: 1.6;
}
.example-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.label-spam { color: #f43f5e; }
.label-ham  { color: #22c55e; }

/* ── Pipeline steps ── */
.pipeline-step {
    background: #fff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 6px;
    font-size: 0.9rem;
    color: #1e293b;
}
.pipeline-arrow {
    text-align: left;
    padding-left: 1.2rem;
    color: #94a3b8;
    font-size: 1rem;
    margin: 2px 0;
}

/* ── Tech badge ── */
.tech-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}

/* ── Sidebar nav ── */
.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.3px;
}
.sidebar-tagline { font-size: 0.78rem; color: #94a3b8; }
.model-info-box {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: 0.82rem;
}

/* ── Table styling ── */
table { width: 100%; border-collapse: collapse; }
th { background: #f8fafc; font-size: 0.82rem; color: #64748b;
     text-transform: uppercase; letter-spacing: 0.5px;
     padding: 8px 12px; border-bottom: 2px solid #e2e8f0; text-align: left; }
td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9;
     font-size: 0.9rem; color: #334155; }
tr:last-child td { border-bottom: none; }
tr.best-row td   { background: #fefce8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading vectorizer ...")
def load_vectorizer():
    if not os.path.exists(VECTOR_PATH):
        return None
    return joblib.load(VECTOR_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)

model      = load_model()
vectorizer = load_vectorizer()
metrics    = load_metrics()


# ── Predict helper ─────────────────────────────────────────────────────────
def predict(text: str):
    cleaned    = clean_text(text)
    vec        = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        confidence = proba[prediction] * 100.0
    else:
        decision   = model.decision_function(vec)[0]
        confidence = (1.0 / (1.0 + np.exp(-abs(decision)))) * 100.0

    label = "SPAM" if prediction == 1 else "HAM"
    return label, round(confidence, 2)


# ── Confidence gauge (Matplotlib) ──────────────────────────────────────────
def confidence_gauge(confidence: float, is_spam: bool):
    """Render a half-donut gauge showing model confidence."""
    fig, ax = plt.subplots(figsize=(3.5, 2.0), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    fill_color = "#f43f5e" if is_spam else "#22c55e"
    bg_color   = "#fee2e2" if is_spam else "#dcfce7"

    theta_full  = np.linspace(np.pi, 0, 200)
    theta_fill  = np.linspace(np.pi, np.pi - (confidence / 100) * np.pi, 200)

    # Background arc
    ax.plot(np.cos(theta_full), np.sin(theta_full),
            color=bg_color, linewidth=18, solid_capstyle="round")
    # Filled arc
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=fill_color, linewidth=18, solid_capstyle="round")

    ax.text(0, -0.15, f"{confidence:.1f}%", ha="center", va="center",
            fontsize=17, fontweight="bold", color=fill_color)
    ax.text(0, -0.52, "Confidence", ha="center", va="center",
            fontsize=8, color="#64748b")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.7, 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ── Inline bar chart for model comparison ──────────────────────────────────
def model_comparison_chart(results: dict, best_name: str):
    names  = list(results.keys())
    f1s    = [results[n]["f1"]        for n in names]
    accs   = [results[n]["accuracy"]  for n in names]
    precs  = [results[n]["precision"] for n in names]
    recs   = [results[n]["recall"]    for n in names]

    x     = np.arange(len(names))
    w     = 0.19
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for i, (vals, lbl, clr) in enumerate(
            zip([accs, precs, recs, f1s],
                ["Accuracy", "Precision", "Recall", "F1 Score"],
                colors)):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lbl,
                      color=clr, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#334155")

    # Highlight best model column
    best_idx = names.index(best_name)
    ax.axvspan(best_idx - 0.42, best_idx + 0.42, color="#fef9c3",
               alpha=0.55, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, fontweight="500")
    ax.set_ylim(0.90, 1.02)
    ax.set_ylabel("Score", fontsize=9, color="#64748b")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.yaxis.grid(True, color="#f1f5f9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0, loc="lower right")
    fig.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
        <div class="sidebar-logo">📧 SpamGuard</div>
        <div class="sidebar-tagline">ML-powered email classifier</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "nav",
        ["🏠  Predict", "📊  Performance", "📁  Dataset", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.divider()

    if metrics:
        best_name_sb = metrics.get("best_model_name", "—")
        bm_sb        = metrics["model_results"].get(best_name_sb, {})
        st.markdown(f"""
        <div class="model-info-box">
            <div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">
                Active Model
            </div>
            <div style="font-size:0.92rem;">{best_name_sb}</div>
            <div style="margin-top:6px; display:flex; gap:12px;">
                <span><b>F1</b> {bm_sb.get('f1','—')}</span>
                <span><b>Acc</b> {bm_sb.get('accuracy','—')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed; bottom:1.5rem; font-size:0.72rem; color:#64748b;">
        Built with Scikit-learn &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Predict
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠  Predict":

    # Resolve best model name dynamically (never hardcode)
    _hero_model = metrics["best_model_name"] if metrics else "ML model"

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">MACHINE LEARNING &nbsp;·&nbsp; NLP &nbsp;·&nbsp; TF-IDF</div>
        <h1>📧 Spam Email Detector</h1>
        <p>Paste any email or message and get an instant SPAM / HAM classification
           powered by a trained <b>{_hero_model}</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Guard
    if model is None or vectorizer is None:
        st.error("Model files not found. Run `python train_model.py` first.")
        st.stop()

    # ── Callbacks: set textarea value BEFORE the widget renders ─────────────
    # Streamlit raises an error if you set widget state after the widget has
    # already been instantiated in the current run.  Using on_click callbacks
    # means the state key is written in the PREVIOUS run, so the textarea
    # picks it up cleanly on the next run.
    def _load_example(text: str):
        st.session_state["email_input_area"] = text

    def _clear_input():
        st.session_state["email_input_area"] = ""

    # ── Input area ──────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-heading">Analyze an Email</div>',
                    unsafe_allow_html=True)

        email_input = st.text_area(
            label="email_input",
            label_visibility="collapsed",
            placeholder="Paste your email or message here ...\n\ne.g.  Congratulations! You've been selected for a FREE $1,000 gift card. Click now!",
            height=175,
            key="email_input_area",
        )

        col_btn, col_clear, col_space = st.columns([1.3, 1, 4.7])
        with col_btn:
            analyse = st.button("🔍  Analyze Email", type="primary",
                                use_container_width=True)
        with col_clear:
            # on_click writes state before the next render — no "can't modify
            # widget after instantiation" error
            st.button("✕  Clear", use_container_width=True, on_click=_clear_input)

    # ── Result ──────────────────────────────────────────────────────────────
    if analyse:
        if not email_input.strip():
            st.warning("Please paste an email or message before clicking Analyze.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    label, confidence = predict(email_input)

                is_spam = label == "SPAM"

                res_col, gauge_col = st.columns([2.2, 1])

                with res_col:
                    if is_spam:
                        st.markdown(f"""
                        <div class="result-spam">
                            <div class="result-emoji">🚨</div>
                            <div class="result-title" style="color:#e11d48;">SPAM DETECTED</div>
                            <div class="result-sub">
                                This message has strong spam characteristics.<br>
                                Do not click links or share personal information.
                            </div>
                            <span class="conf-pill conf-pill-spam">
                                Model confidence: {confidence:.2f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-ham">
                            <div class="result-emoji">✅</div>
                            <div class="result-title" style="color:#16a34a;">LEGITIMATE EMAIL</div>
                            <div class="result-sub">
                                This message appears to be genuine ham mail.<br>
                                No spam signals were detected.
                            </div>
                            <span class="conf-pill conf-pill-ham">
                                Model confidence: {confidence:.2f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                with gauge_col:
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_gauge = confidence_gauge(confidence, is_spam)
                    st.pyplot(fig_gauge, use_container_width=True)
                    plt.close(fig_gauge)

            except Exception as exc:
                st.error(f"Prediction error: {exc}")

    # ── Examples ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Try an Example</div>',
                unsafe_allow_html=True)

    EXAMPLES = [
        {
            "type"  : "SPAM",
            "short" : "Prize winner",
            "text"  : "Congratulations! You have won a $1,000 Walmart gift card. "
                      "Click the link below to claim your FREE prize before it expires!",
        },
        {
            "type"  : "SPAM",
            "short" : "Urgent offer",
            "text"  : "URGENT: Your account has been suspended. Verify your details "
                      "immediately at http://secure-login-verify.com or lose access forever.",
        },
        {
            "type"  : "HAM",
            "short" : "Work email",
            "text"  : "Hi, can you please send me the updated project report before "
                      "tomorrow's 10 am meeting? Thanks in advance.",
        },
        {
            "type"  : "HAM",
            "short" : "Meeting invite",
            "text"  : "Hey, are you free on Thursday afternoon for a quick catch-up call? "
                      "I'd like to walk you through the Q3 results.",
        },
    ]

    ex_cols = st.columns(4)
    for col, ex in zip(ex_cols, EXAMPLES):
        is_s = ex["type"] == "SPAM"
        lbl_cls  = "label-spam" if is_s else "label-ham"
        lbl_text = "SPAM EXAMPLE" if is_s else "HAM EXAMPLE"
        with col:
            st.markdown(f"""
            <div class="example-card">
                <div class="example-label {lbl_cls}">{lbl_text} &mdash; {ex['short']}</div>
                {ex['text'][:120]}{'...' if len(ex['text']) > 120 else ''}
            </div>
            """, unsafe_allow_html=True)
            # on_click writes directly to the textarea key so the text
            # appears immediately on the next render — no copy-paste step needed.
            st.button(
                "Use this example",
                key=f"ex_{ex['short']}",
                use_container_width=True,
                on_click=_load_example,
                args=(ex["text"],),
            )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Performance
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊  Performance":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">EVALUATION &nbsp;·&nbsp; METRICS &nbsp;·&nbsp; COMPARISON</div>
        <h1>📊 Model Performance</h1>
        <p>Accuracy, precision, recall and F1-score across all three trained classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics is None:
        st.error("Metrics not found. Run `python train_model.py` first.")
        st.stop()

    best_name    = metrics["best_model_name"]
    results      = metrics["model_results"]
    dataset_info = metrics.get("dataset_info", {})
    bm           = results[best_name]

    # ── Dataset tiles ──────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Dataset Split</div>',
                unsafe_allow_html=True)

    tile_cols = st.columns(5)
    tile_data = [
        ("Total",    dataset_info.get("total", 0), "#3b82f6"),
        ("Ham",      dataset_info.get("ham",   0), "#22c55e"),
        ("Spam",     dataset_info.get("spam",  0), "#f43f5e"),
        ("Training", dataset_info.get("train", 0), "#8b5cf6"),
        ("Testing",  dataset_info.get("test",  0), "#f59e0b"),
    ]
    for col, (lbl, val, clr) in zip(tile_cols, tile_data):
        col.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value" style="color:{clr};">{val:,}</div>
            <div class="stat-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Best model metrics ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-heading">Best Model: {best_name}</div>',
                unsafe_allow_html=True)

    m_cols = st.columns(4)
    m_data = [
        ("Accuracy",  bm.get("accuracy",  0), "#3b82f6"),
        ("Precision", bm.get("precision", 0), "#8b5cf6"),
        ("Recall",    bm.get("recall",    0), "#22c55e"),
        ("F1 Score",  bm.get("f1",        0), "#f59e0b"),
    ]
    for col, (lbl, val, clr) in zip(m_cols, m_data):
        col.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value" style="color:{clr};">{val:.4f}</div>
            <div class="stat-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Precision / Recall explainer ───────────────────────────────────────
    with st.expander("Why Precision & Recall matter in spam filtering"):
        st.markdown("""
**Precision** — of all emails flagged as spam, how many actually were?
A low precision means legitimate emails are wrongly junked (false positives).

**Recall** — of all real spam emails, how many did the model catch?
A low recall means spam reaches the inbox (false negatives).

**F1 Score** — harmonic mean of precision and recall; the primary model-selection metric here.

> **False positives are especially costly**: a legitimate email (job offer, bank alert)
> wrongly labelled as spam can have serious real-world consequences.
""")

    # ── Model comparison chart ─────────────────────────────────────────────
    st.markdown('<div class="section-heading">Model Comparison</div>',
                unsafe_allow_html=True)

    fig_cmp = model_comparison_chart(results, best_name)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    # ── Comparison table ───────────────────────────────────────────────────
    rows_html = ""
    for name, m in results.items():
        is_best   = name == best_name
        row_class = ' class="best-row"' if is_best else ""
        marker    = " &#9733; Best" if is_best else ""
        cv_f1_val = m.get("cv_f1", "—")
        cv_f1_str = f"{cv_f1_val:.4f}" if isinstance(cv_f1_val, float) else cv_f1_val
        rows_html += f"""
        <tr{row_class}>
            <td><b>{name}</b>{marker}</td>
            <td>{cv_f1_str}</td>
            <td>{m['accuracy']:.4f}</td>
            <td>{m['precision']:.4f}</td>
            <td>{m['recall']:.4f}</td>
            <td><b>{m['f1']:.4f}</b></td>
        </tr>"""

    st.markdown(f"""
    <div class="card" style="padding:0; overflow:hidden;">
    <table>
        <thead><tr>
            <th>Model</th>
            <th title="5-fold CV F1 on training data — used for model selection">CV F1 (selection)</th>
            <th>Accuracy</th><th>Precision</th><th>Recall</th>
            <th>Test F1</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    <p style="font-size:0.78rem; color:#64748b; margin-top:6px;">
        Model selected by <b>CV F1</b> (5-fold on training data only) to avoid
        optimistic bias from choosing the winner on the same test set used for
        final reporting.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix ───────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Confusion Matrix</div>',
                unsafe_allow_html=True)

    cm_left, cm_right = st.columns([1, 1.6])
    with cm_left:
        if os.path.exists(CM_IMG):
            st.image(CM_IMG, use_container_width=True)
        else:
            st.info("Run train_model.py to generate the confusion matrix.")

    with cm_right:
        st.markdown("""
        <div style="padding-top:1rem;">
        <table>
            <thead><tr><th></th><th>Predicted HAM</th><th>Predicted SPAM</th></tr></thead>
            <tbody>
            <tr>
                <td><b>Actual HAM</b></td>
                <td style="color:#16a34a;"><b>True Negative (TN)</b><br>
                    <span style="color:#64748b;font-size:0.82rem;">
                    Correctly identified as legitimate</span></td>
                <td style="color:#dc2626;"><b>False Positive (FP)</b><br>
                    <span style="color:#64748b;font-size:0.82rem;">
                    Legit email wrongly flagged as spam</span></td>
            </tr>
            <tr>
                <td><b>Actual SPAM</b></td>
                <td style="color:#d97706;"><b>False Negative (FN)</b><br>
                    <span style="color:#64748b;font-size:0.82rem;">
                    Spam that slipped through</span></td>
                <td style="color:#16a34a;"><b>True Positive (TP)</b><br>
                    <span style="color:#64748b;font-size:0.82rem;">
                    Correctly caught spam</span></td>
            </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dataset
# ══════════════════════════════════════════════════════════════════════════
elif page == "📁  Dataset":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">DATA &nbsp;·&nbsp; DISTRIBUTION &nbsp;·&nbsp; INSIGHTS</div>
        <h1>📁 Dataset Insights</h1>
        <p>Explore the spam/ham dataset used to train the classifier.</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics is None:
        st.error("Metrics not found. Run `python train_model.py` first.")
        st.stop()

    dataset_info = metrics.get("dataset_info", {})
    total = dataset_info.get("total", 0)
    spam  = dataset_info.get("spam",  0)
    ham   = dataset_info.get("ham",   0)

    # ── Summary tiles ──────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Overview</div>',
                unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.columns(5)
    tiles = [
        (t1, "Total Emails",   total, "#3b82f6"),
        (t2, "Ham Emails",     ham,   "#22c55e"),
        (t3, "Spam Emails",    spam,  "#f43f5e"),
        (t4, "Ham Ratio",      f"{ham/total*100:.1f}%" if total else "—", "#8b5cf6"),
        (t5, "Spam Ratio",     f"{spam/total*100:.1f}%" if total else "—", "#f59e0b"),
    ]
    for col, lbl, val, clr in tiles:
        col.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value" style="color:{clr}; font-size:1.7rem;">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Class Distribution</div>',
                unsafe_allow_html=True)

    dist_left, dist_right = st.columns([1, 1.4])

    with dist_left:
        if os.path.exists(DIST_IMG):
            st.image(DIST_IMG, use_container_width=True)
        else:
            fig_d, ax_d = plt.subplots(figsize=(4, 3))
            ax_d.bar(["HAM", "SPAM"], [ham, spam],
                     color=["#22c55e", "#f43f5e"], width=0.45,
                     edgecolor="white")
            ax_d.set_ylabel("Count", fontsize=9)
            ax_d.spines[["top", "right"]].set_visible(False)
            fig_d.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)

    with dist_right:
        # Inline donut via matplotlib
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3.2))
        fig_pie.patch.set_facecolor("#ffffff")
        wedge_sizes  = [ham, spam]
        wedge_colors = ["#22c55e", "#f43f5e"]
        wedges, texts, autotexts = ax_pie.pie(
            wedge_sizes,
            labels=["Ham", "Spam"],
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10),
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
            at.set_color("white")
        ax_pie.set_title("Ham vs Spam Ratio", fontsize=11, pad=8, color="#1e293b")
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset details card ───────────────────────────────────────────────
    st.markdown('<div class="section-heading">Dataset Details</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>Source file:</b> <code>data/spam_ham_dataset.csv</code><br><br>
        <b>Columns used:</b>
        <table style="margin-top:8px;">
            <thead><tr><th>Column</th><th>Description</th></tr></thead>
            <tbody>
            <tr><td><code>text</code></td>
                <td>Raw email body (includes subject line)</td></tr>
            <tr><td><code>label</code></td>
                <td>Human-readable label: <code>ham</code> or <code>spam</code></td></tr>
            <tr><td><code>label_num</code></td>
                <td>Numeric encoding: 0 = ham, 1 = spam</td></tr>
            </tbody>
        </table>
    </div>
    <div class="card" style="margin-top:0.5rem;">
        <b>No synthetic data was added.</b>
        The model was trained exclusively on this dataset.
        <br><br>
        <b>Class imbalance:</b> The dataset is realistically imbalanced (~71% ham, ~29% spam).
        Stratified train/test splitting ensures both splits preserve this ratio.
        <br><br>
        <b>Preprocessing applied:</b> lowercase conversion, whitespace normalisation.
        URLs, numbers, and punctuation are <em>kept</em> because they carry important
        spam signal for TF-IDF.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">PROJECT &nbsp;·&nbsp; PIPELINE &nbsp;·&nbsp; TECHNOLOGIES</div>
        <h1>ℹ️ About This Project</h1>
        <p>A complete end-to-end supervised ML application built entirely in Python.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Project Overview</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    This application uses <b>supervised machine learning</b> to classify email messages
    as <b>spam</b> or <b>legitimate (ham)</b>.<br><br>
    Three classical text classifiers are trained on TF-IDF features extracted from real
    email data. The best-performing model — selected by F1-score — is persisted and used
    for all live predictions.<br><br>
    The entire workflow from raw CSV to real-time web predictions is implemented in
    Python using only open-source libraries. No external APIs, no pre-trained language
    models, and no synthetic data.
    </div>
    """, unsafe_allow_html=True)

    # ── ML pipeline ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">ML Pipeline</div>',
                unsafe_allow_html=True)

    steps = [
        ("1", "Load dataset",               "CSV with 5,171 labelled emails"),
        ("2", "Text cleaning",               "Lowercase, whitespace normalisation"),
        ("3", "Train / test split",          "80 / 20, stratified by class"),
        ("4", "TF-IDF feature extraction",   "Fit on train only — zero data leakage"),
        ("5", "Train three classifiers",     "Logistic Regression, Naive Bayes, Linear SVM"),
        ("6", "Evaluate on test set",        "Accuracy, Precision, Recall, F1"),
        ("7", "Select best model",           "By highest F1-score"),
        ("8", "Save model + vectorizer",     "Joblib — loaded at app startup"),
        ("9", "Live prediction",             "User text → cleaned → TF-IDF → model → result"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="background:#dbeafe;color:#1d4ed8;border-radius:50%;
                         display:inline-block;width:22px;height:22px;text-align:center;
                         font-size:0.75rem;font-weight:700;line-height:22px;
                         margin-right:8px;">{num}</span>
            <b>{title}</b>
            <span style="color:#64748b; font-size:0.85rem;"> — {desc}</span>
        </div>
        """, unsafe_allow_html=True)
        if num != "9":
            st.markdown('<div class="pipeline-arrow">&#8595;</div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Technologies ───────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Technologies</div>',
                unsafe_allow_html=True)

    tech_list = [
        ("Python 3",      "Core language"),
        ("Pandas",        "Data loading & cleaning"),
        ("NumPy",         "Numerical ops"),
        ("Scikit-learn",  "TF-IDF, ML models, metrics"),
        ("Matplotlib",    "Visualisations"),
        ("Streamlit",     "Web app UI"),
        ("Joblib",        "Model serialisation"),
    ]
    badges = " ".join(
        f'<span class="tech-badge" title="{desc}">{name}</span>'
        for name, desc in tech_list
    )
    st.markdown(f'<div style="margin-bottom:1rem;">{badges}</div>',
                unsafe_allow_html=True)

    # ── Why these models ───────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Why These Models?</div>',
                unsafe_allow_html=True)

    model_rows = ""
    for name, why in [
        ("Logistic Regression",
         "Strong linear baseline with well-calibrated class probabilities."),
        ("Multinomial Naive Bayes",
         "Classic bag-of-words classifier; fast and surprisingly effective on TF-IDF."),
        ("Linear SVM",
         "Maximises the margin in high-dimensional TF-IDF space; typically strongest on text."),
    ]:
        model_rows += f"<tr><td><b>{name}</b></td><td>{why}</td></tr>"

    st.markdown(f"""
    <div class="card" style="padding:0; overflow:hidden;">
    <table>
        <thead><tr><th>Model</th><th>Rationale</th></tr></thead>
        <tbody>{model_rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Future improvements ────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Future Improvements</div>',
                unsafe_allow_html=True)

    improvements = [
        "Advanced NLP: stemming, lemmatisation, entity removal",
        "Larger and more diverse email corpora",
        "Email header analysis (sender IP, SPF / DKIM validation)",
        "Ensemble / stacking classifiers",
        "Deep learning: LSTM or BERT-based transformer",
        "Real inbox integration via IMAP",
        "Docker containerisation and cloud deployment",
    ]
    items_html = "".join(
        f'<li style="margin-bottom:6px; color:#334155; font-size:0.9rem;">{item}</li>'
        for item in improvements
    )
    st.markdown(f"""
    <div class="card">
        <ul style="padding-left:1.2rem; margin:0;">{items_html}</ul>
    </div>
    """, unsafe_allow_html=True)
