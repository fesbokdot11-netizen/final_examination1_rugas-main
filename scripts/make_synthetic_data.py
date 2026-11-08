"""Generate a SYNTHETIC stand-in for student_service_requests.csv.

400 rows, 14 columns, 327 completed + 73 open/cancelled (146 null cells), mirroring the
shape of the midterm dataset. REPLACE data/student_service_requests.csv with your real
midterm CSV (if authorized for public use) and re-run scripts/train_model.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "student_service_requests.csv"

PLAN = [  # service type, count, office, base SLA days
    ("Records", 85, "Registrar", 5),
    ("Certification", 72, "Registrar", 3),
    ("Scholarship Inquiry", 47, "Scholarship Office", 5),
    ("Enrollment Assistance", 70, "Student Affairs", 4),
    ("Finance Clearance", 66, "Finance", 5),
    ("ID Replacement", 60, "Student Affairs", 7),
]
rows = []
for name, n, office, sla in PLAN:
    rows += [(name, office, sla)] * n
df = pd.DataFrame(rows, columns=["Service_Type", "Assigned_Office", "base_sla"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
N = len(df)  # 400

df["Request_ID"] = [f"REQ-2026-{i + 1:04d}" for i in range(N)]
start = pd.Timestamp("2026-01-05")
df["Submission_Date"] = [(start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                         for d in rng.integers(0, 270, N)]
df["Requester_Type"] = rng.choice(["Student", "Faculty", "Staff"], N, p=[0.7, 0.15, 0.15])
df["Student_Program"] = rng.choice(["BSIS", "BSCS", "BSIT", "BSBA", "BSED"], N)
df["Channel"] = rng.choice(["Online", "Walk-in"], N, p=[0.65, 0.35])
df["Priority"] = rng.choice(["Normal", "High"], N, p=[0.8, 0.2])
df["Queue_Length"] = np.clip(rng.poisson(10, N), 0, 30)
df["SLA_Days"] = np.maximum(2, df["base_sla"] + rng.choice([-1, 0, 0, 1], N))
df["Required_Documents_Complete"] = rng.choice(["Yes", "No"], N, p=[0.75, 0.25])

# status: exactly 73 non-completed (9 of them Scholarship Inquiry so 38 stay completed)
sch_idx = df.index[df["Service_Type"] == "Scholarship Inquiry"].to_numpy()
oth_idx = df.index[df["Service_Type"] != "Scholarship Inquiry"].to_numpy()
noncomp = np.concatenate([rng.choice(sch_idx, 9, replace=False), rng.choice(oth_idx, 64, replace=False)])
df["Status"] = "Completed"
open_part = rng.permutation(noncomp)
df.loc[open_part[:50], "Status"] = "Open"
df.loc[open_part[50:], "Status"] = "Cancelled"

# breach process (depends only on submission-time factors + noise)
office_eff = {"Scholarship Office": 1.6, "Finance": 2.2, "Registrar": -0.5, "Student Affairs": -0.3}
weekday = pd.to_datetime(df["Submission_Date"]).dt.day_name()
z = (0.6 + 0.30 * (df["Queue_Length"] - 10)
     + np.where(df["Required_Documents_Complete"] == "Yes", -1.5, 1.5)
     - 0.60 * (df["SLA_Days"] - 5)
     + df["Assigned_Office"].map(office_eff)
     + np.where(weekday.isin(["Monday", "Friday"]), 0.3, 0.0))
prob = 1 / (1 + np.exp(-z))
breach = (rng.random(N) < prob).astype(int)

# force the Scholarship Office finding from the midterm: 30 breaches / 38 completed
comp = df["Status"] == "Completed"
sch_comp = df.index[comp & (df["Assigned_Office"] == "Scholarship Office")].to_numpy()
cur = breach[sch_comp].sum()
if cur > 30:
    flip = rng.choice(sch_comp[breach[sch_comp] == 1], cur - 30, replace=False)
    breach[flip] = 0
elif cur < 30:
    flip = rng.choice(sch_comp[breach[sch_comp] == 0], 30 - cur, replace=False)
    breach[flip] = 1

proc = np.where(breach == 1,
                df["SLA_Days"] + rng.integers(1, 7, N),
                [rng.integers(1, s + 1) for s in df["SLA_Days"]])
sat = np.clip(np.round(rng.normal(np.where(breach == 1, 2.8, 4.1), 0.8)), 1, 5)
df["Processing_Days"] = np.where(comp, proc, np.nan)
df["Satisfaction_Score"] = np.where(comp, sat, np.nan)

cols = ["Request_ID", "Submission_Date", "Service_Type", "Requester_Type", "Student_Program",
        "Channel", "Assigned_Office", "Priority", "Queue_Length", "SLA_Days",
        "Required_Documents_Complete", "Status", "Processing_Days", "Satisfaction_Score"]
df[cols].to_csv(OUT, index=False)
print(f"wrote {OUT} rows={len(df)} cols={len(cols)} nulls={int(df[cols].isna().sum().sum())}")
