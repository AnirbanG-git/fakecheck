# 🧠 FakeCheck — A Hybrid Retrieval and Verification Pipeline for Automated Fact Checking

**Author:** Anirban Gangopadhyay  
**Institution:** Liverpool John Moores University (MS Thesis 2025)  
**Supervisor:** Monica
**Tag:** `v1.0-thesis-final` (Final implementation version)

---

## 📘 Overview

**FakeCheck** is an explainable, hybrid **Retrieval-Augmented Verification (RAV)** system designed to automatically validate factual claims.  
It integrates **lexical**, **semantic**, and **transformer-based verification** components to detect fake or misleading claims using evidence retrieved from a large corpus.

### 🔍 Motivation

With the explosive growth of misinformation, manual fact-checking has become unsustainable.  
FakeCheck automates this by:
1. Retrieving relevant evidence for a claim,  
2. Verifying factual consistency using transformer models, and  
3. Generating human-readable explanations with supporting citations.

---

## ⚙️ System Architecture

The FakeCheck pipeline consists of three modular agents:

```mermaid
graph TD
    A[Claim Input] --> B[Retriever Node<br>(BM25 + E5 Embeddings)]
    B --> C[Verifier Node<br>(DistilBERT Classifier)]
    C --> D[Explainer Node<br>(Evidence-based Rationale)]
    D --> E[Final Verdict + Confidence + Explanation]
```

Each module is self-contained and reusable within the LangChain or LangGraph framework.

---

## 🧩 Components

| Module | Description |
|:-------|:-------------|
| **Retriever** | Combines **BM25** (lexical) and **DenseRetriever (E5 embeddings)** using Reciprocal Rank Fusion (RRF). |
| **Verifier** | Fine-tuned **DistilBERT** model trained on claim–evidence pairs from WELFake dataset. |
| **Explainer** | Produces human-readable justifications linking claims to supporting or contradicting snippets. |
| **Dataset** | [WELFake dataset](https://huggingface.co/datasets/davanstrien/WELFake) — ~72k news samples labelled as fake or genuine. |

---

## 🧪 Reproduction Instructions

### 1️⃣ Setup

```bash
git clone https://github.com/AnirbanG-git/fakecheck.git
cd fakecheck
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure you have GPU access if available (CUDA or ROCm).

---

### 2️⃣ Run the pipeline for a single claim

```bash
PYTHONPATH=. python scripts/run_agent_cli.py \
  --claim "141 people were arrested at a protest of the Dakota Access Pipeline." \
  --mode bert \
  --csv data/WELFake.csv \
  --threshold 0.45 \
  --dense_index_dir indexes/welfake_dense \
  --out reports/demo_claim.json
```

Output example (`reports/demo_claim.json`):
```json
{
  "claim": "141 people were arrested at a protest of the Dakota Access Pipeline.",
  "verdict": 1,
  "confidence": 0.79,
  "explanation": "Verdict: Real (confidence: high)...",
  "evidence_internal": [...],
  "evidence_external": [...]
}
```

---

### 3️⃣ Batch evaluation (25 test claims)

```bash
PYTHONPATH=. python scripts/run_25_from_csv_titles.py
```

Output: `reports/week5_graph_run.jsonl`

---

### 4️⃣ Evaluation & Comparison (Week 6)

```bash
PYTHONPATH=. python scripts/evaluate_week6.py \
  --pred reports/week5_graph_run.jsonl \
  --truth_csv data/WELFake.csv \
  --out_dir reports/week6

PYTHONPATH=. python scripts/compare_models_week6.py \
  --pipeline_pred reports/week5_graph_run.jsonl \
  --labels_jsonl reports/verifier_dataset.jsonl \
  --truth_csv data/WELFake.csv \
  --lr_artifact artifacts/verifier_lr/baseline_tfidf_lr.joblib \
  --bert_dir artifacts/verifier_bert \
  --out_dir reports/week6
```

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|:------|:---------:|:---------:|:------:|:--:|:-----:|
| **Baseline (LR)** | 0.68 | 0.62 | **1.00** | 0.76 | 0.87 |
| **Standalone BERT** | 0.52 | 0.52 | **1.00** | 0.68 | 0.76 |
| **Hybrid Pipeline** | **0.72** | **0.88** | 0.54 | 0.67 | **0.90** |

> Average latency: 389 ms/claim (GPU)  
> Confirms best precision–recall balance and highest AUROC among models.

---

## 🧩 Project Folder Structure

```
fakecheck/
│
├── src/
│   ├── agent/               # LangGraph/agent logic
│   │   ├── nodes.py         # Retrieve, Verify, Explain nodes
│   │   ├── state.py         # Data structures
│   │   └── ...
│   ├── retriever/           # BM25, Dense, and hybrid retrieval modules
│   └── utils/               # Query expansion, preprocessing
│
├── scripts/
│   ├── run_agent_cli.py     # Single-claim CLI
│   ├── run_25_from_csv_titles.py
│   ├── evaluate_week6.py
│   ├── compare_models_week6.py
│   ├── error_analysis_week6.py
│   └── make_week6_notes.py
│
├── reports/
│   ├── week1–6/             # JSON, plots, and summaries
│   └── demo_claim.json
│
├── artifacts/
│   ├── verifier_bert/       # Fine-tuned DistilBERT model
│   ├── verifier_lr/         # TF-IDF + LR baseline
│   └── indexes/
│
├── data/
│   └── WELFake.csv          # Reference dataset (public)
│
└── requirements.txt
```

---

## 🧭 Week-wise Progress

| Week | Focus | Key Outputs |
|:--|:--|:--|
| Week 1 | EDA & Baseline (TF-IDF + LR) | `metrics_baseline.json` (Acc ≈ 0.95) |
| Week 2–3 | Retriever (BM25, Dense) | Corpus encoding, RRF fusion |
| Week 4 | Verifier (DistilBERT Fine-tuning) | `verifier_bert` artifact |
| Week 5 | Integration (LangGraph pipeline) | `week5_graph_run.jsonl`, `demo_claim.json` |
| Week 6 | Final Evaluation + Visualization | `week6/final_metrics.json`, ROC & confusion matrix plots |

---

## 💡 Key Insights

- **Hybrid retrieval** improves contextual matching and robustness.  
- **Verifier fine-tuning** increases AUROC from 0.86 → 0.90.  
- **Explainability** via evidence-linked text improves trust and interpretability.  
- Final latency is acceptable for practical deployments.

---

## 📚 Citation

If you refer to this project:

```
@thesis{gangopadhyay2025fakecheck,
  title={FakeCheck: A Hybrid Retrieval and Verification Pipeline for Automated Fact Checking},
  author={Anirban Gangopadhyay},
  year={2025},
  school={Liverpool John Moores University},
}
```

---

## 🏁 Status

✅ Implementation complete (Week 6)  
🧾 Ready for thesis writing and submission  
🏷️ Version: **v1.0-thesis-final**

---

## 📬 Contact

**Anirban Gangopadhyay**  
GitHub: [@AnirbanG-git](https://github.com/AnirbanG-git)  
Email: [anir.dr@gmail.com]
