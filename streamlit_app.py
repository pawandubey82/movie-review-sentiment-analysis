"""Streamlit dashboard for the movie-review sentiment classifier.

Run:  python -m streamlit run streamlit_app.py
"""
import html
import math
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_preprocessing import clean_review
from src.predict import load_model

POS, NEG, ACC = "#10b981", "#f43f5e", "#6366f1"
st.set_page_config(page_title="Cinescope", page_icon="🎬", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
[data-testid="stMetric"]{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px 16px}
mark{border-radius:5px;padding:2px 3px;color:inherit}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    best = load_model()
    allm = joblib.load(config.ALL_MODELS_PATH) if config.ALL_MODELS_PATH.exists() else {}
    return best, allm


def prob_positive(model, text):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    return 1 / (1 + math.exp(-2 * float(model.decision_function([text])[0])))


def explain(model, text, top=8):
    """Word contribution = tf-idf weight x model coefficient."""
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["classifier"]
    X = tfidf.transform([text]).tocoo()
    w = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    names = tfidf.get_feature_names_out()
    items = sorted(((names[j], float(v * w[j])) for j, v in zip(X.col, X.data)), key=lambda t: t[1])
    return items, [i for i in items if i[1] < 0][:top], [i for i in reversed(items) if i[1] > 0][:top]


def style(fig, h=320):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def highlight(raw, items):
    w = {n: c for n, c in items if " " not in n}
    mx = max([abs(c) for c in w.values()] + [1e-3])

    def mark(m):
        c = w.get(m.group(0).lower())
        if not c:
            return m.group(0)
        a = min(0.45, abs(c) / mx * 0.45)
        rgb = "16,185,129" if c > 0 else "244,63,94"
        return f'<mark style="background:rgba({rgb},{a:.2f})">{m.group(0)}</mark>'

    return re.sub(r"[A-Za-z']+", mark, html.escape(raw))


def clear():
    st.session_state.review, st.session_state.result = "", None


# ---------------- setup ----------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)

st.title("🎬 Review Sentiment Analytics")
st.caption("Classify movie reviews, see which words drove the decision, and compare model performance.")

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python main.py` first, then reload this page.")
    st.stop()
best_model, all_models = get_models()

df = pd.read_csv(config.COMPARISON_PATH) if config.COMPARISON_PATH.exists() else None
k = st.columns(5)
if df is not None:
    b = df.loc[df["F1 Score"].idxmax()]
    k[0].metric("Best model", b["Model"])
    k[1].metric("Accuracy", f"{b['Accuracy']*100:.1f}%")
    k[2].metric("F1 score", f"{b['F1 Score']:.3f}")
    k[3].metric("ROC-AUC", f"{b['ROC-AUC']:.3f}")
k[4].metric("Analyzed this session", len(st.session_state.history))

tab1, tab2 = st.tabs(["Analyze", "Model performance"])

# ---------------- analyze tab ----------------
with tab1:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Review input")
        raw = st.text_area("Paste a movie review", key="review", height=200,
                           placeholder="A beautiful, moving film with brilliant performances...",
                           max_chars=10000)
        c1, c2, _ = st.columns([1, 1, 2])
        run = c1.button("Analyze", type="primary", use_container_width=True)
        c2.button("Clear", on_click=clear, use_container_width=True)
        st.caption(f"{len(raw.split())} words · {len(raw)} characters")

    if run:
        if not raw.strip():
            st.warning("Please enter a review first.")
        else:
            text = clean_review(raw)
            p = prob_positive(best_model, text)
            items, neg, pos = explain(best_model, text)
            votes = {n: prob_positive(m, text) for n, m in all_models.items()}
            st.session_state.result = dict(raw=raw, p=p, items=items, neg=neg, pos=pos,
                                           votes=votes, words=len(text.split()))
            st.session_state.history.append(p)
            st.rerun()

    r = st.session_state.result
    with right:
        st.subheader("Prediction")
        if not r:
            st.info("Analyze a review to see results.")
        else:
            positive = r["p"] >= 0.5
            col = POS if positive else NEG
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=r["p"] * 100, number={"suffix": "%"},
                title={"text": "Probability of positive"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
                       "steps": [{"range": [0, 50], "color": "rgba(244,63,94,.15)"},
                                 {"range": [50, 100], "color": "rgba(16,185,129,.15)"}]}))
            st.plotly_chart(style(g, 250))
            conf = r["p"] if positive else 1 - r["p"]
            st.markdown(f"<h2 style='color:{col};margin:0'>{'Positive' if positive else 'Negative'}</h2>"
                        f"<span style='color:#8b97ab'>{conf*100:.1f}% confident · {r['words']} words</span>",
                        unsafe_allow_html=True)

    if r:
        st.divider()
        a, b2 = st.columns(2, gap="large")
        with a:
            st.subheader("Word attribution")
            st.caption("Green words push toward positive, red toward negative.")
            it = sorted(r["pos"] + r["neg"], key=lambda t: t[1])
            fig = go.Figure(go.Bar(x=[i[1] for i in it], y=[i[0] for i in it], orientation="h",
                                   marker_color=[POS if i[1] > 0 else NEG for i in it]))
            st.plotly_chart(style(fig, 340))
        with b2:
            st.subheader("Model consensus")
            if r["votes"]:
                names, vals = list(r["votes"]), [v * 100 for v in r["votes"].values()]
                fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[POS if v >= 50 else NEG for v in vals],
                                       text=[f"{v:.1f}%" for v in vals], textposition="outside"))
                fig.add_hline(y=50, line_dash="dot", line_color="#8b97ab")
                fig.update_yaxes(range=[0, 110], title="P(positive) %")
                st.plotly_chart(style(fig, 340))
        st.subheader("Highlighted review")
        st.markdown(highlight(r["raw"], r["items"]), unsafe_allow_html=True)

    if len(st.session_state.history) > 1:
        st.subheader("Session timeline")
        fig = go.Figure(go.Scatter(y=[h * 100 for h in st.session_state.history], mode="lines+markers",
                                   line=dict(color=ACC), fill="tozeroy", fillcolor="rgba(99,102,241,.14)"))
        fig.update_yaxes(range=[0, 100], title="P(positive) %")
        fig.update_xaxes(title="Review #")
        st.plotly_chart(style(fig, 220))

# ---------------- performance tab ----------------
with tab2:
    if df is None:
        st.info("Run `python main.py` to generate model_comparison.csv.")
    else:
        cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        pal = [ACC, POS, "#f59e0b", "#38bdf8"]
        fig = go.Figure()
        for i, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(r=[row[c] for c in cols] + [row[cols[0]]],
                                          theta=cols + [cols[0]], name=row["Model"],
                                          line=dict(color=pal[i % 4]), fill="toself", opacity=0.55))
        fig.update_polars(radialaxis=dict(range=[0.8, 1.0]))
        st.plotly_chart(style(fig, 420))
        st.dataframe(df, hide_index=True)

st.caption("TF-IDF features + classical ML, trained on 50K IMDb reviews. Sarcasm and very short text can fool the model.")