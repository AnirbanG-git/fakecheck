import json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())

def main():
    report = {
        "baseline_lr": load("reports/metrics_verifier_lr.json"),
        "verifier_bert": load("reports/metrics_verifier_bert.json"),
        "coverage_external": load("reports/coverage_external_sweep_adapted.json"),
        "explanation_eval": load("reports/eval_explanations.json"),
    }
    Path("reports/week4_summary.json").write_text(json.dumps(report, indent=2))
    print("Wrote combined summary → reports/week4_summary.json")

if __name__ == "__main__":
    main()
