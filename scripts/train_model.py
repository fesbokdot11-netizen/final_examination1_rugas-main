"""Train and save the preprocessing + Decision Tree pipeline (run ONCE, offline).

Usage (from repo root):  python scripts/train_model.py [--csv data/student_service_requests.csv]
Reproduces the midterm setup: completed requests only, 80/20 split, random_state=42,
OneHotEncoder + DecisionTreeClassifier in one scikit-learn Pipeline.
If your midterm used different hyper-parameters or features, edit CONFIG below to match.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules import data_processing as dp  # noqa: E402
from modules.prediction import CAT_FEATURES, FEATURE_COLUMNS, NUM_FEATURES  # noqa: E402

CONFIG = {"test_size": 0.20, "random_state": 42, "max_depth": 4, "min_samples_leaf": 5}


def main(csv_path: Path):
    raw = pd.read_csv(csv_path)
    errors, _ = dp.validate_dataframe(raw)
    if errors:
        raise SystemExit("Dataset invalid: " + " | ".join(errors))
    df = dp.prepare_dataframe(raw)
    lab = dp.labeled(df)
    X, y = lab[FEATURE_COLUMNS], lab["SLA_Breached"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    pipe = Pipeline([
        ("prep", pre),
        ("clf", DecisionTreeClassifier(max_depth=CONFIG["max_depth"],
                                       min_samples_leaf=CONFIG["min_samples_leaf"],
                                       random_state=CONFIG["random_state"])),
    ])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "majority_baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())),
        "fp_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    # aggregate one-hot importances back to the original feature names
    names = pipe.named_steps["prep"].get_feature_names_out()
    imps = pipe.named_steps["clf"].feature_importances_
    agg = {f: 0.0 for f in FEATURE_COLUMNS}
    for n, v in zip(names, imps):
        if n.startswith("num__"):
            agg[n[5:]] += float(v)
        else:
            for f in CAT_FEATURES:
                if n.startswith(f"cat__{f}_"):
                    agg[f] += float(v)
                    break
    importances = dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))

    breach_by_office = lab.groupby("Assigned_Office")["SLA_Breached"].mean()
    meta = {
        "model_version": "1.0.0",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "DecisionTreeClassifier in sklearn Pipeline (OneHotEncoder + passthrough numerics)",
        "sklearn_version": sklearn.__version__,
        "training_data": str(csv_path.name),
        "config": CONFIG,
        "features": FEATURE_COLUMNS,
        "n_labeled": int(len(lab)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "metrics": metrics,
        "feature_importances": importances,
        "categories": {c: sorted(lab[c].dropna().unique().tolist())
                       for c in ["Service_Type", "Requester_Type", "Assigned_Office"]},
        "high_breach_offices": breach_by_office[breach_by_office >= 0.70].index.tolist(),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "trained_model.joblib")
    (ROOT / "models" / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["n_labeled", "n_train", "n_test", "metrics", "feature_importances"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "student_service_requests.csv"))
    main(Path(ap.parse_args().csv))
