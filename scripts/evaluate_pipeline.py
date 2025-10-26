#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def load_jsonl(p):
    rows=[]
    with open(p,"r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reports/verifier_dataset.jsonl")
    ap.add_argument("--graph_preds", default="reports/week5_graph_run.jsonl")
    ap.add_argument("--verifier_preds_lr", default="reports/preds_verifier_lr.jsonl")
    ap.add_argument("--verifier_preds_bert", default="reports/preds_verifier_bert.jsonl")
    ap.add_argument("--out", default="reports/final_metrics.json")
    args = ap.parse_args()

    ds = {r["id"]: r for r in load_jsonl(args.dataset) if r.get("split")=="test"}
    ids = sorted(ds.keys())
    y_true = [ds[i]["label"] for i in ids]

    def to_map(path, kind):
        if not Path(path).exists(): return {}
        m={}
        for r in load_jsonl(path):
            rid = r.get("id")
            if rid in ds:
                if kind=="graph":
                    m[rid] = {"label": r.get("verdict"), "proba": r.get("confidence")}
                else:
                    m[rid] = {"label": r.get("pred_label", r.get("label")), "proba": r.get("pred_proba", r.get("proba"))}
        return m

    gmap = to_map(args.graph_preds, "graph")
    lmap = to_map(args.verifier_preds_lr, "lr")
    bmap = to_map(args.verifier_preds_bert, "bert")

    def vec(m):
        yp, pp = [], []
        for i in ids:
            r = m.get(i)
            yp.append(int(r["label"]) if r and r.get("label") is not None else 0)
            pp.append(float(r["proba"]) if r and r.get("proba") is not None else 0.5)
        return yp, pp

    def pack(y_pred, p_score):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            auroc = roc_auc_score(y_true, p_score)
        except Exception:
            auroc = None
        return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auroc":auroc}

    out = {
        "n_test": len(ids),
        "graph": pack(*vec(gmap)) if gmap else None,
        "verifier_lr": pack(*vec(lmap)) if lmap else None,
        "verifier_bert": pack(*vec(bmap)) if bmap else None
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
