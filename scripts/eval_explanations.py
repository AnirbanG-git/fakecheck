import argparse, json
from pathlib import Path
from statistics import mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="reports/eval_explanations.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    total = len(rows)

    verdict_counts = {}
    avg_len = mean(len(r.get("explanation","")) for r in rows)
    avg_cits = mean(len(r.get("citations",[])) for r in rows)
    empty_cits = sum(1 for r in rows for c in r.get("citations",[]) if not c.get("url")) 
    conf_by_verdict = {}

    for r in rows:
        v = r.get("verdict","?")
        verdict_counts[v] = verdict_counts.get(v,0)+1
        conf_by_verdict.setdefault(v, []).append(r.get("proba",0))

    out = {
        "n_explanations": total,
        "verdict_counts": verdict_counts,
        "avg_explanation_length": round(avg_len,2),
        "avg_citations_per_claim": round(avg_cits,2),
        "percent_empty_urls": round(empty_cits/(total*avg_cits+1e-6)*100,2),
        "mean_confidence_by_verdict": {k: round(mean(v),3) for k,v in conf_by_verdict.items()},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
