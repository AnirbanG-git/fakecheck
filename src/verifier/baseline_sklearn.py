from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_recall_fscore_support
)
import joblib

def train_tfidf_lr(X_train, y_train):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, prob)
    else:
        auroc = float("nan")

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    report = classification_report(y_test, preds, digits=4)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc)
    }
    return metrics, report
