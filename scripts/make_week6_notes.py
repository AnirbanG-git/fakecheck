#!/usr/bin/env python3
import json
from pathlib import Path

FINAL = Path("reports/week6/final_metrics.json")
CMP = Path("reports/week6/model_comparison.csv")
OUT = Path("reports/week6/notes_week6.md")

def main():
    if FINAL.exists():
        m = json.loads(FINAL.read_text())
    else:
        m = {}
    lines = ["# Week 6 – Results & Discussion Notes", ""]
    if m:
        lines += [
            "## Pipeline Metrics",
            f"- N = {m.get('n')}",
            f"- Accuracy = {m.get('accuracy'):.3f}",
            f"- Precision = {m.get('precision'):.3f}",
            f"- Recall = {m.get('recall'):.3f}",
            f"- F1 = {m.get('f1'):.3f}",
            f"- AUROC = {m.get('auroc'):.3f}",
            "",
        ]
        lat = m.get("latency_ms_summary") or {}
        if lat.get("count"):
            lines += [
                "## Latency",
                f"- mean = {lat.get('mean'):.1f} ms, median = {lat.get('median'):.1f} ms, p90 = {lat.get('p90'):.1f} ms",
                ""
            ]
    if CMP.exists():
        lines += [
            "## Model Comparison",
            "See `reports/week6/model_comparison.csv` for full table.",
            ""
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__=="__main__":
    main()

