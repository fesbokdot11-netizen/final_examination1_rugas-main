# Student Service Request Monitoring - SLA Breach Decision Support (Streamlit)

Final laboratory exam, *Systems Analysis and Design* (BS Information Systems, Section B).
Deploys the midterm **Decision Tree + OneHotEncoder** pipeline that predicts whether a student
service request will breach its service-level agreement (SLA), inside a human-in-the-loop workflow.

> Predictions are decision-support outputs. They never authorize rejecting a request or denying a student service.

## Modules (sidebar)
| Page | What it does |
|---|---|
| Home | Purpose, usage steps, model summary |
| Dataset | CSV upload, schema checks, preview, null/duplicate summary |
| Data Analysis | Descriptive stats, class distribution, 5 interactive Plotly charts |
| Prediction | One-row inference, probability, drivers, limitations, metrics vs midterm, batch scoring |
| Dashboard | Responsive KPI cards, searchable/sortable risk queue and request register, workflow-status badges, interpretation, role-based final decision log |
| System Design | 5 functional + 3 non-functional requirements, actors, use case diagram, workflow |
| Presentation | Timed presenter script with page-by-page demo cues |

## Repository layout
```
app.py                      Streamlit entry point
requirements.txt
modules/                    data_processing.py, exploratory_analysis.py, prediction.py, dashboard.py
models/trained_model.joblib + model_metadata.json (saved pipeline and metrics; never retrained on refresh)
assets/use_case_diagram.png
data/student_service_requests.csv   SYNTHETIC sample data
scripts/                    make_synthetic_data.py, train_model.py, make_use_case_diagram.py
docs/deployment_report_template.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use YOUR midterm model and data (important)
The bundled model was trained on synthetic data shaped like the midterm dataset. To reuse your real work:
1. In your midterm notebook, save the fitted pipeline (preprocessing + model) and make sure it was trained on
   exactly the columns in `modules/prediction.py::FEATURE_COLUMNS` (edit that list if your features differ):
   ```python
   import joblib; joblib.dump(pipeline, "trained_model.joblib")
   ```
2. Put it in `models/trained_model.joblib`. Pin `scikit-learn` in `requirements.txt` to the version used to save it
   (`import sklearn; print(sklearn.__version__)`).
3. Alternatively, put your CSV in `data/` and run `python scripts/train_model.py --csv data/yourfile.csv`
   (edit `CONFIG` to match your midterm hyper-parameters). This also refreshes `models/model_metadata.json`.
4. Do **not** commit real student records. Use synthetic or properly authorized data for public deployment.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (`git init`, `git add .`, `git commit`, `git push`).
2. Open https://share.streamlit.io and sign in with GitHub.
3. Create app -> choose repository and branch -> main file path `app.py` -> Deploy.
4. If the build fails, open the app logs (Manage app) and check `requirements.txt` versions.

## Leakage controls
Model inputs are submission-time fields only. `Processing_Days`, `Satisfaction_Score`, `Status`, `SLA_Breached`
and `Request_ID` are never passed to the pipeline. Training uses completed requests only (open/cancelled
requests have no outcome).

The dashboard maps workflow states to Pending, Processing, Completed, or Delayed badges. `Processing` is
accepted as an optional in-progress status; the bundled sample currently uses Open, Completed, and Cancelled.

## Known limitations
See the Limitations panel on the Prediction page and `docs/deployment_report.md`. Update the report after
publishing the app and completing the remaining notebook-comparison and upload-validation tests.
