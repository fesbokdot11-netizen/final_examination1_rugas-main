# Deployment Report - Section B (fill in before submitting)

**Student:** ____________  **GitHub repo:** ____________  **Streamlit URL:** ____________

## 1. Reused midterm code
- Data loading / cleaning: ...
- Feature engineering (Queue_Category, Submission_Weekday, SLA_Breached): ...
- Model (Decision Tree, 80/20, random_state=42, OneHotEncoder pipeline): ...

## 2. Changes from the midterm
- Saved the fitted pipeline with joblib; app never retrains on refresh.
- Differences between the midterm table/problem and the deployed app (explain honestly): ...

## 3. Tests performed
| Test | Expected | Result |
|---|---|---|
| Valid CSV upload | Preview + quality summary | |
| CSV missing a required column | Clear error naming the column | |
| Empty CSV | Error, no crash | |
| Valid prediction input | Label, probability, tier, timestamp | |
| Invalid input (e.g. unseen category) | Error message | |
| Same test cases as midterm notebook | Same predictions | |
| Administrator tries to record decision | Blocked (read-only) | |

## 4. Limitations
(see Prediction page) ...

## 5. Reflection
...
