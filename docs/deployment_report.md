# Deployment Report - Section B

**Student:** Kristheal Sofia Rugas  **GitHub repo:** https://github.com/fesbokdot11-netizen/final_examination1_rugas
**Streamlit URL:** [not deployed yet]

## 1. Reused midterm code
- Data loading / cleaning: `modules/data_processing.py`
- Feature engineering: `Queue_Category`, `Submission_Weekday`, and `SLA_Breached` logic in `modules/data_processing.py`
- Model: Decision Tree pipeline with OneHotEncoder and passthrough numerical features from `scripts/train_model.py`
- Inference and scoring: `modules/prediction.py`

## 2. Changes from the midterm
- The fitted pipeline is saved as `models/trained_model.joblib` and loaded with `@st.cache_resource` so it is not retrained on every refresh.
- The app wraps the model in a human-in-the-loop decision-support workflow instead of exposing raw model output alone.
- The dashboard and system-design pages translate the technical model into an information-system workflow for staff review.
- The saved model was regenerated with scikit-learn 1.9.1 for compatibility with the current project environment.
- Important difference: the current bundled dataset is synthetic. Confirm the features, training split, and reported
  metrics against the submitted midterm notebook before claiming exact model continuity.

## 3. Tests performed
| Test | Expected | Result |
|---|---|---|
| Sample CSV loads and preprocessing completes | 400 rows loaded and derived features present | Passed |
| Model loads and accepts a valid one-row input | Prediction returns a class and probabilities | Passed |
| Streamlit Home and Dashboard render | No app exceptions in Streamlit smoke tests | Passed |
| `Processing` is accepted as a workflow status | Validation returns no errors | Passed |
| Invalid upload, missing columns, and empty CSV | Clear validation errors | Not yet verified end-to-end |
| Same cases compared with the submitted midterm notebook | Matching predictions | Not yet verified; original notebook cases were unavailable |
| Administrator decision recording is blocked | Read-only role behavior | Not yet verified end-to-end |
| Public Streamlit deployment | URL opens successfully | Not deployed yet |

## 4. Limitations
- Predictions are advisory only and do not authorize denial of service.
- The model is trained on completed requests only, because open/cancelled requests do not have an outcome label.
- The Decision Tree is shallow and the probability estimates are coarse.
- The demo dataset is synthetic and must not be used publicly with real student data.

## 5. Reflection
The application structure converts the SLA prediction workflow into a decision-support experience with validation,
analysis, and human review. Before submission, compare the bundled model against the original midterm notebook and
replace the synthetic sample/model with the authorized midterm artifacts if they differ.
