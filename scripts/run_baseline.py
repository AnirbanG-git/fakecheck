import argparse, json
from pathlib import Path
from src.utils.data_io import load_welfake
from src.utils.split import stratified_splits
from src.verifier.baseline_sklearn import train_tfidf_lr, evaluate
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to WELFake.csv")
    ap.add_argument("--out_dir", required=True, help="Directory to save model artifacts")
    ap.add_argument("--report", required=True, help="JSON path to write metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    args = ap.parse_args()

    df = load_welfake(args.csv)

    # Concatenate title + text for the baseline input
    text = (df.get("title", "") + " " + df.get("text", "")).astype(str)
    df = df.assign(_text=text)

    df_train, df_val, df_test = stratified_splits(
        df, seed=args.seed, test_size=args.test_size, val_size=args.val_size
    )

    X_train, y_train = df_train["_text"].tolist(), df_train["label"].tolist()
    X_val,   y_val   = df_val["_text"].tolist(),   df_val["label"].tolist()
    X_test,  y_test  = df_test["_text"].tolist(),  df_test["label"].tolist()

    model = train_tfidf_lr(X_train, y_train)

    # Evaluate on test
    metrics, report = evaluate(model, X_test, y_test)

    # Save artifacts and metrics
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(args.out_dir) / "baseline_tfidf_lr.joblib")

    Path(Path(args.report).parent).mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(Path(args.report).with_suffix(".txt"), "w") as f:
        f.write(report)

    print("Saved model to", Path(args.out_dir) / "baseline_tfidf_lr.joblib")
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
