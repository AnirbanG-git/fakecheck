import streamlit as st
from pathlib import Path
import json
from typing import Any
from pydantic import BaseModel

from src.agent.state import AgentState, ClaimInput
from src.agent.nodes import RetrieveNode, VerifyNode, ExplainNode
from src.agent.graph import build_graph

# ---------- utils ----------
def to_plain(x: Any):
    """Recursively convert Pydantic models (and nested structures) to plain Python types."""
    if isinstance(x, BaseModel):
        return x.model_dump()
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    return x

@st.cache_resource
def load_corpus(csv_path="data/WELFake.csv", text_col="text", id_col="id"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    meta = []
    if id_col in df.columns:
        for v in df[id_col].tolist():
            try:
                meta.append({"id": int(v)})
            except Exception:
                meta.append({})
    else:
        meta = [{} for _ in range(len(df))]
    return texts, meta

@st.cache_resource
def build_app(top_k_internal: int, dense_index_dir: str = "artifacts/e5_base", mode: str = "bert"):
    texts, meta = load_corpus()
    retrieve = RetrieveNode(texts, meta, top_k_internal=top_k_internal, dense_index_dir=dense_index_dir)
    verify = VerifyNode(mode=mode)
    explain = ExplainNode()
    graph = build_graph(retrieve, verify, explain)
    return graph

# ---------- UI ----------
st.set_page_config(page_title="FakeCheck Agent", layout="wide")
st.title("🧠 FakeCheck Agent — Retrieve → Verify → Explain")

claim = st.text_area("Enter a claim/article snippet:", height=120, placeholder="Type a claim…")
col1, col2, col3 = st.columns(3)
with col1:
    mode = st.selectbox("Verifier", ["bert", "lr"], index=0)
with col2:
    topk = st.slider("Internal top-k", 3, 10, 5)
with col3:
    run = st.button("Run fact-check")

# Build the app (cached)
graph = build_app(top_k_internal=topk, mode=mode)

if "history" not in st.session_state:
    st.session_state["history"] = []

if run and claim.strip():
    # Build initial state as dict for LangGraph
    init_state = AgentState(claim=ClaimInput(text=claim.strip())).model_dump()
    out_state = graph.invoke(init_state)     # dict (may contain pydantic from nodes)
    out = to_plain(out_state)                # normalize to plain dict

    result = {
        "claim": out.get("claim", {}).get("text", claim.strip()),
        "label": int(out.get("verdict", {}).get("label", 0)),
        "proba": float(out.get("verdict", {}).get("proba", 0.0)),
        "explanation": out.get("verdict", {}).get("explanation", ""),
        "internal": [
            {
                "text": e.get("text",""),
                "score": float(e.get("score", 0.0)),
                "meta": e.get("meta", {}),
            }
            for e in out.get("evidence", {}).get("internal", [])
        ],
        "external": out.get("evidence", {}).get("external", []),
        "trace_ms": out.get("trace", {}),
    }

    st.session_state["history"].append(result)

st.subheader("Results")
for i, r in enumerate(reversed(st.session_state["history"]), 1):
    st.markdown(f"### #{i}. Verdict: **{'Real' if r['label']==1 else 'Fake'}** (p={r['proba']:.2f})")
    st.markdown(r["explanation"])
    with st.expander("Internal evidence"):
        for j, e in enumerate(r["internal"], 1):
            st.write(f"**{j}.** score={e['score']:.3f}  meta={e['meta']}")
            st.write(e["text"])
    with st.expander("External evidence"):
        if r["external"]:
            for j, e in enumerate(r["external"], 1):
                st.write(f"**{j}.** [{e.get('source','')}] {e.get('url','')}")
                st.write(e.get("snippet",""))
        else:
            st.write("_none_")
    # Optional: download
    st.download_button(
        f"Download result #{i}",
        data=json.dumps(r, ensure_ascii=False, indent=2),
        file_name=f"factcheck_result_{i}.json",
        mime="application/json",
        key=f"dl_{i}",
    )
    st.write("---")
