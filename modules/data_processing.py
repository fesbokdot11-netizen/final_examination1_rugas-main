"""Dataset loading, schema validation, cleaning and feature engineering.

Mirrors the midterm preprocessing:
  * SLA_Breached = 1 if Processing_Days > SLA_Days (completed requests only)
  * Queue_Category: Low (0-5), Medium (6-15), High (>15)
  * Submission_Weekday = Submission_Date.dt.day_name()
Streamlit is optional here so scripts/train_model.py can reuse this module.
"""
from pathlib import Path

import pandas as pd

try:  # cache safe, repeated transformations when running inside Streamlit
    import streamlit as st

    _cache = st.cache_data(show_spinner=False)
except Exception:  # training script / tests without streamlit
    def _cache(func):
        return func

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "student_service_requests.csv"

REQUIRED_COLUMNS = [
    "Request_ID", "Submission_Date", "Service_Type", "Requester_Type",
    "Assigned_Office", "Priority", "Queue_Length", "SLA_Days",
    "Required_Documents_Complete", "Status", "Processing_Days",
]
OPTIONAL_COLUMNS = ["Satisfaction_Score", "Student_Program", "Channel"]
NUMERIC_COLUMNS = ["Queue_Length", "SLA_Days", "Processing_Days"]
VALID_STATUS = ["Completed", "Open", "Processing", "Cancelled"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
QUEUE_ORDER = ["Low", "Medium", "High"]


def bin_queue(n) -> str:
    """Low (0-5), Medium (6-15), High (>15)."""
    if n <= 5:
        return "Low"
    if n <= 15:
        return "Medium"
    return "High"


def normalize_yes_no(series: pd.Series) -> pd.Series:
    mapping = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes",
               "no": "No", "n": "No", "false": "No", "0": "No", "0.0": "No"}
    return series.astype(str).str.strip().str.lower().map(mapping)


def validate_dataframe(df: pd.DataFrame):
    """Return (errors, warnings). Errors block use of the file; warnings do not."""
    errors, warnings = [], []
    if df is None or df.empty:
        return ["The file is empty (no data rows). Upload a CSV with at least one request."], warnings

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing) +
                      ". Expected columns: " + ", ".join(REQUIRED_COLUMNS) + ".")
        return errors, warnings  # cannot check further safely

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna() & df[col].notna()
        if bad.any():
            errors.append(f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
        elif (converted < 0).any():
            errors.append(f"Column '{col}' contains negative values.")

    for col in ["Queue_Length", "SLA_Days"]:
        if df[col].isna().any():
            errors.append(f"Column '{col}' has {int(df[col].isna().sum())} missing value(s); it is a model input.")

    dates = pd.to_datetime(df["Submission_Date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"Column 'Submission_Date' has {int(dates.isna().sum())} missing or unparsable date(s).")

    status = df["Status"].astype(str).str.strip().str.title()
    bad_status = sorted(set(status) - set(VALID_STATUS))
    if bad_status:
        errors.append("Unknown Status value(s): " + ", ".join(bad_status) +
                      ". Allowed: " + ", ".join(VALID_STATUS) + ".")

    if normalize_yes_no(df["Required_Documents_Complete"]).isna().any():
        errors.append("Column 'Required_Documents_Complete' must contain Yes/No (or True/False, 1/0).")

    if df["Request_ID"].duplicated().any():
        warnings.append(f"{int(df['Request_ID'].duplicated().sum())} duplicate Request_ID value(s) found.")
    if df.duplicated().any():
        warnings.append(f"{int(df.duplicated().sum())} fully duplicated row(s) found.")

    completed = status.eq("Completed")
    if completed.any() and df.loc[completed, "Processing_Days"].isna().any():
        warnings.append("Some Completed requests have no Processing_Days; they cannot be used as labeled records.")
    return errors, warnings


def data_quality_summary(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "total_nulls": int(nulls.sum()),
        "nulls_by_column": nulls[nulls > 0].rename("Null cells").to_frame(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df.columns else None,
    }


def prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean + engineer features. Assumes validate_dataframe() returned no errors."""
    df = raw.copy()
    df["Status"] = df["Status"].astype(str).str.strip().str.title()
    df["Required_Documents_Complete"] = normalize_yes_no(df["Required_Documents_Complete"])
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    numeric = NUMERIC_COLUMNS + (["Satisfaction_Score"] if "Satisfaction_Score" in df.columns else [])
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    df["Queue_Category"] = df["Queue_Length"].apply(bin_queue)

    labeled_mask = df["Status"].eq("Completed") & df["Processing_Days"].notna()
    df["SLA_Breached"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[labeled_mask, "SLA_Breached"] = (
        df.loc[labeled_mask, "Processing_Days"] > df.loc[labeled_mask, "SLA_Days"]
    ).astype(int)
    return df


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Completed requests with a known outcome (only rows valid for training / breach EDA)."""
    out = df[df["SLA_Breached"].notna()].copy()
    out["SLA_Breached"] = out["SLA_Breached"].astype(int)
    return out


@_cache
def load_sample() -> pd.DataFrame:
    return prepare_dataframe(pd.read_csv(SAMPLE_PATH))


@_cache
def prepare_cached(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_dataframe(raw)
