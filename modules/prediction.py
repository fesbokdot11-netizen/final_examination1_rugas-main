"""Model loading helpers, input validation, one-row inference and batch scoring.

The saved scikit-learn Pipeline (OneHotEncoder + DecisionTreeClassifier) is trained
offline by scripts/train_model.py - it is NEVER retrained on a page refresh.
Only submission-time features are used. Processing_Days, Satisfaction_Score,
Status and SLA_Breached are excluded to avoid outcome leakage.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from modules.data_processing import bin_queue

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

CAT_FEATURES = ["Service_Type", "Requester_Type", "Assigned_Office",
                "Required_Documents_Complete", "Submission_Weekday", "Queue_Category"]
NUM_FEATURES = ["Queue_Length", "SLA_Days"]
FEATURE_COLUMNS = CAT_FEATURES + NUM_FEATURES
LEAKAGE_COLUMNS = ["Processing_Days", "Satisfaction_Score", "Status", "SLA_Breached", "Request_ID"]

HIGH_RISK_THRESHOLD = 0.70
FLAG_THRESHOLD = 0.50

# Results reported in the submitted midterm (Part IV), shown next to the deployed model.
MIDTERM_REPORTED = {
    "Accuracy": 0.682,
    "Precision (Breach=1)": 0.651,
    "Recall (Breach=1)": 0.824,
    "F1 (Breach=1)": 0.727,
    "Confusion matrix": {"TN": 17, "FP": 15, "FN": 6, "TP": 28},
    "Majority baseline accuracy": 0.515,
}

LIMITATIONS = [
    "Decision-support only: a flag never authorizes denying or rejecting a service request.",
    "Trained on completed requests only; open/cancelled requests have no known outcome.",
    "Small training set and a shallow Decision Tree, so probabilities are coarse, not calibrated.",
    "The midterm false-positive rate was about 46.8% on non-breach cases, so many flags will be false alarms.",
    "Only submission-time features are used; later events (staff absence, system downtime) are unseen.",
    "Demo data is synthetic. Do not deploy publicly with real student records.",
]


def load_metadata() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_file():
    return joblib.load(MODEL_PATH)


def validate_inputs(inputs: dict, meta: dict) -> list:
    errors = []
    cats = meta.get("categories", {})
    for field in ["Service_Type", "Requester_Type", "Assigned_Office"]:
        if inputs.get(field) in (None, ""):
            errors.append(f"{field} is required.")
        elif cats.get(field) and inputs[field] not in cats[field]:
            errors.append(f"{field} '{inputs[field]}' was not seen in training data.")
    if inputs.get("Required_Documents_Complete") not in ("Yes", "No"):
        errors.append("Required_Documents_Complete must be Yes or No.")
    try:
        q = int(inputs.get("Queue_Length"))
        if q < 0 or q > 200:
            errors.append("Queue_Length must be between 0 and 200.")
    except (TypeError, ValueError):
        errors.append("Queue_Length must be a whole number.")
    try:
        s = int(inputs.get("SLA_Days"))
        if s < 1 or s > 60:
            errors.append("SLA_Days must be between 1 and 60.")
    except (TypeError, ValueError):
        errors.append("SLA_Days must be a whole number.")
    if inputs.get("Submission_Date") is None:
        errors.append("Submission_Date is required.")
    return errors


def build_input_row(inputs: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact training feature names, order and types."""
    date = pd.to_datetime(inputs["Submission_Date"])
    queue = int(inputs["Queue_Length"])
    row = {
        "Service_Type": str(inputs["Service_Type"]),
        "Requester_Type": str(inputs["Requester_Type"]),
        "Assigned_Office": str(inputs["Assigned_Office"]),
        "Required_Documents_Complete": str(inputs["Required_Documents_Complete"]),
        "Submission_Weekday": date.day_name(),
        "Queue_Category": bin_queue(queue),
        "Queue_Length": queue,
        "SLA_Days": int(inputs["SLA_Days"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _breach_proba(model, X: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if 1 not in classes:
        return None
    return model.predict_proba(X)[:, classes.index(1)]


def risk_tier(prob) -> str:
    if prob is None or prob != prob:  # None or NaN
        return "Unknown"
    if prob >= HIGH_RISK_THRESHOLD:
        return "High"
    if prob >= FLAG_THRESHOLD:
        return "Elevated"
    return "Low"


def predict_single(model, row: pd.DataFrame, meta: dict) -> dict:
    label = int(model.predict(row)[0])
    proba = _breach_proba(model, row)
    p = float(proba[0]) if proba is not None else None
    context = []
    if row["Queue_Category"].iloc[0] == "High":
        context.append("High queue backlog at submission (more than 15 open requests).")
    if row["Required_Documents_Complete"].iloc[0] == "No":
        context.append("Required documents are not complete.")
    if row["Assigned_Office"].iloc[0] in meta.get("high_breach_offices", []):
        context.append(f"{row['Assigned_Office'].iloc[0]} has a historically high breach rate.")
    return {
        "label": label,
        "label_text": "Likely SLA breach" if label == 1 else "Likely within SLA",
        "probability": p,
        "tier": risk_tier(p),
        "context": context,
        "model_version": meta.get("model_version", "unknown"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def recommended_action(prob, docs_complete: str, queue_cat: str) -> str:
    tier = risk_tier(prob)
    if tier == "High":
        steps = ["Review for reassignment / fast-track"]
    elif tier == "Elevated":
        steps = ["Monitor and confirm staffing"]
    else:
        steps = ["Standard queue"]
    if docs_complete == "No":
        steps.append("request missing documents")
    if queue_cat == "High" and tier in ("High", "Elevated"):
        steps.append("rebalance office workload")
    return "; ".join(steps)


def score_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Add Breach_Probability, Risk_Tier, Predicted_Breach_Flag and Recommended_Action."""
    out = df.copy()
    if out.empty:
        for c in ["Predicted_Breach_Flag", "Breach_Probability", "Risk_Tier", "Recommended_Action"]:
            out[c] = pd.Series(dtype="object")
        return out
    X = out[FEATURE_COLUMNS]
    out["Predicted_Breach_Flag"] = model.predict(X).astype(int)
    proba = _breach_proba(model, X)
    out["Breach_Probability"] = proba if proba is not None else float("nan")
    out["Risk_Tier"] = [risk_tier(p) for p in out["Breach_Probability"]]
    out["Recommended_Action"] = [
        recommended_action(p, d, q) for p, d, q in
        zip(out["Breach_Probability"], out["Required_Documents_Complete"], out["Queue_Category"])
    ]
    return out
