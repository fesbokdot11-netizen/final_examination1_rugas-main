"""Student Service Request Monitoring - SLA breach decision-support system (Streamlit)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules import dashboard as dash
from modules import data_processing as dp
from modules import exploratory_analysis as eda
from modules import prediction as pred

st.set_page_config(page_title="SLA Breach Decision Support", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --navy: #142b4a;
        --blue: #2f67a5;
        --soft-blue: #eaf2fb;
        --ink: #172b45;
        --muted: #64748b;
        --line: #dce5ef;
    }
    .stApp { background: #f5f8fc; color: var(--ink); }
    .block-container {
        max-width: 1440px;
        padding: 1.5rem 2rem 2.5rem;
    }
    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #203e61;
    }
    [data-testid="stSidebar"] * { color: #f8fbff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        transition: background-color 140ms ease, color 140ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #203e61; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
        background: #dcecff;
        color: var(--navy) !important;
        font-weight: 700;
    }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 8px rgba(20, 43, 74, 0.04);
    }
    .metric-card {
        background: #fff;
        border: 1px solid var(--line);
        border-top: 3px solid var(--blue);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(20, 43, 74, 0.04);
        margin-bottom: 1rem;
        min-height: 112px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--navy);
        margin-bottom: 0.25rem;
    }
    .metric-meta { font-size: 0.8rem; color: var(--muted); }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--navy);
        margin: 1.4rem 0 0.7rem;
    }
    .alert-box {
        background: var(--soft-blue);
        border: 1px solid #c8dbef;
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: var(--navy);
        margin: 0.5rem 0 1rem;
    }
    .home-hero {
        background: linear-gradient(135deg, #142b4a 0%, #2f67a5 100%);
        border-radius: 18px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 12px 24px rgba(20, 43, 74, 0.12);
    }
    .home-hero h1 {
        color: #ffffff !important;
        margin: 0 0 0.45rem;
        font-size: 2.4rem;
    }
    .home-hero p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1rem;
        margin: 0;
    }
    .home-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        font-weight: 700;
    }
    .home-card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 18px rgba(20, 43, 74, 0.06);
        height: 100%;
    }
    .home-card h4 {
        color: var(--navy);
        margin: 0 0 0.45rem;
        font-size: 1rem;
    }
    .home-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.55;
    }
    .home-step {
        background: #f7fafe;
        border: 1px solid #dfeaf9;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        height: 100%;
    }
    .home-step .num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.8rem;
        height: 1.8rem;
        border-radius: 50%;
        background: #dcecff;
        color: var(--navy);
        font-weight: 800;
        margin-bottom: 0.7rem;
    }
    .home-step h5 {
        margin: 0 0 0.3rem;
        color: var(--navy);
    }
    .home-step p {
        margin: 0;
        color: var(--muted);
        font-size: 0.93rem;
    }
    .decision-note {
        background: #fff9ea;
        border: 1px solid #f4de9d;
        border-left: 4px solid #d99d22;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #5b4305;
        margin-top: 1rem;
        font-weight: 600;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 10px; }
    .stButton > button, [data-testid="stDownloadButton"] button {
        border-radius: 8px;
        border: 1px solid #b9cde2;
        transition: background-color 140ms ease, border-color 140ms ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background: var(--soft-blue);
        border-color: var(--blue);
    }
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.8rem 1.5rem; }
        .metric-card { min-height: 96px; padding: 0.8rem; }
        .metric-value { font-size: 1.55rem; }
        h1 { font-size: 1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design", "Presentation"]
PAGE_ICONS = {
    "Home": "⌂",
    "Dataset": "▤",
    "Data Analysis": "◷",
    "Prediction": "◎",
    "Dashboard": "▦",
    "System Design": "◇",
    "Presentation": "▶",
}


@st.cache_resource
def load_model():
    return pred.load_model_file()


@st.cache_data
def load_meta():
    return pred.load_metadata()


def get_active_df():
    if "active_df" in st.session_state:
        return st.session_state["active_df"], st.session_state["active_name"]
    return dp.load_sample(), "Bundled synthetic sample"


def model_or_stop():
    try:
        return load_model(), load_meta()
    except Exception as exc:  # missing/incompatible joblib
        st.error("The trained model could not be loaded. Run `python scripts/train_model.py` and make sure "
                 "scikit-learn matches requirements.txt.")
        st.exception(exc)
        st.stop()


# ----------------------------------------------------------------- sidebar
st.sidebar.title("SLA Monitoring")
st.sidebar.caption("STUDENT SERVICES")
menu = st.sidebar.radio(
    "Navigation",
    PAGES,
    format_func=lambda page: f"{PAGE_ICONS[page]}  {page}",
    label_visibility="collapsed",
)
role = st.sidebar.selectbox("Acting role (demo)", dash.ROLES,
                            help="Service Officers can record final decisions; Administrators are read-only.")
active_df, active_name = get_active_df()
st.sidebar.caption(f"Active data: **{active_name}** ({len(active_df):,} rows)")


# ----------------------------------------------------------------- pages
def page_home():
    st.markdown(
        """
        <div class="home-hero">
            <div class="home-badge">Student Services</div>
            <h1>Student Service Request Monitoring</h1>
            <p>Track backlog risk, flag SLA breaches early, and support staff decisions with a transparent, human-in-the-loop workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta = None
    try:
        meta = load_meta()
    except Exception:
        st.warning("Model metadata not found. Run scripts/train_model.py.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Requests loaded", f"{len(active_df):,}")
    c2.metric("Model version", meta["model_version"] if meta else "n/a")
    c3.metric("Test accuracy (deployed)", f"{meta['metrics']['accuracy']:.1%}" if meta else "n/a")

    st.markdown("### What this dashboard helps with")
    cols = st.columns(3)
    cards = [
        ("Backlog visibility", "Identify which request queues and offices are accumulating delays before SLA targets are missed."),
        ("Risk detection", "Use the deployed decision-tree model to highlight service requests with a higher probability of SLA breach."),
        ("Human review", "Keep final decisions with staff so the system supports action rather than replacing judgment."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="home-card">
                <h4>{title}</h4>
                <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### How to use the system")
    steps = st.columns(5)
    step_items = [
        ("Dataset", "Upload a CSV or use the bundled sample to validate the request data."),
        ("Analysis", "Review volume, service bottlenecks, and breach patterns across offices."),
        ("Prediction", "Score a request and inspect the model’s risk estimate and drivers."),
        ("Dashboard", "Monitor KPIs, queue risk, and final decision status in one view."),
        ("Workflow", "Use the system design information to align the tool with operational policy."),
    ]
    for col, (title, body) in zip(steps, step_items):
        with col:
            st.markdown(f"""
            <div class="home-step">
                <div class="num">{step_items.index((title, body)) + 1}</div>
                <h5>{title}</h5>
                <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="decision-note">
            Predictions are decision-support outputs. They do not authorize rejecting or denying a service request, and the demo data is synthetic.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_quality(raw: pd.DataFrame):
    q = dp.data_quality_summary(raw)
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{q['rows']:,}")
    b.metric("Columns", q["columns"])
    c.metric("Null cells", f"{q['total_nulls']:,}")
    d.metric("Duplicate rows", q["duplicate_rows"])
    if q["duplicate_ids"] is not None:
        st.caption(f"Duplicate Request_ID values: {q['duplicate_ids']}")
    if q["total_nulls"]:
        st.markdown("**Null cells by column**")
        st.dataframe(q["nulls_by_column"], use_container_width=True)
        st.caption("Nulls in Processing_Days / Satisfaction_Score are expected for open or cancelled requests.")
    else:
        st.success("No null cells detected.")


def page_dataset():
    st.header("Dataset Upload and Validation")
    st.markdown("Required columns: `" + "`, `".join(dp.REQUIRED_COLUMNS) + "`.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    st.download_button("Download sample template (synthetic CSV)", dp.SAMPLE_PATH.read_bytes(),
                       "student_service_requests_sample.csv", "text/csv")

    if uploaded is None:
        st.info("No file uploaded. Showing the bundled synthetic sample.")
        raw = pd.read_csv(dp.SAMPLE_PATH)
        show_quality(raw)
        st.dataframe(raw.head(20), use_container_width=True)
        return

    try:
        raw = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        st.error("The file is empty or has no header row.")
        return
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"The file could not be read as a CSV: {exc}")
        return

    errors, warnings = dp.validate_dataframe(raw)
    st.subheader("Preview")
    st.dataframe(raw.head(20), use_container_width=True)
    st.subheader("Data quality")
    if not raw.empty:
        show_quality(raw)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    st.success("Schema validation passed.")
    if st.button("Use this dataset across the app"):
        st.session_state["active_df"] = dp.prepare_cached(raw)
        st.session_state["active_name"] = uploaded.name
        st.rerun()


def page_prediction():
    st.header("Predictive Modeling")
    model, meta = model_or_stop()
    tab1, tab2, tab3 = st.tabs(["Predict one request", "Model metrics", "Score current dataset"])

    with tab1:
        st.markdown("Enter **submission-time** information only. Outcome fields (Processing_Days, "
                    "Satisfaction_Score, Status) are not accepted, so nothing leaks into inference.")
        cats = meta["categories"]
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            service = c1.selectbox("Service type", cats["Service_Type"])
            requester = c2.selectbox("Requester type", cats["Requester_Type"])
            office = c3.selectbox("Assigned office", cats["Assigned_Office"])
            c4, c5, c6, c7 = st.columns(4)
            queue = c4.number_input("Queue length at submission", min_value=0, max_value=200, value=12, step=1)
            sla = c5.number_input("SLA days", min_value=1, max_value=60, value=5, step=1)
            docs = c6.radio("Required documents complete?", ["Yes", "No"], horizontal=True)
            date = c7.date_input("Submission date")
            go = st.form_submit_button("Predict SLA breach risk")
        if go:
            inputs = {"Service_Type": service, "Requester_Type": requester, "Assigned_Office": office,
                      "Queue_Length": queue, "SLA_Days": sla, "Required_Documents_Complete": docs,
                      "Submission_Date": date}
            errs = pred.validate_inputs(inputs, meta)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                row = pred.build_input_row(inputs)
                res = pred.predict_single(model, row, meta)
                st.markdown("#### Result")
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", res["label_text"])
                m2.metric("Breach probability",
                          f"{res['probability']:.0%}" if res["probability"] is not None else "not supported")
                m3.metric("Risk tier", res["tier"])
                if res["tier"] in ("High", "Elevated"):
                    st.warning("Flag for **human review** by student services staff. This is not a denial.")
                else:
                    st.success("Low predicted risk. Standard handling; staff may still review.")
                for line in res["context"]:
                    st.markdown(f"- {line}")
                imp = pd.Series(meta["feature_importances"]).head(3)
                st.markdown("**Top model drivers (global):** " +
                            ", ".join(f"{k} ({v:.2f})" for k, v in imp.items()))
                st.caption(f"Model v{res['model_version']} | predicted at {res['timestamp']} | "
                           "advisory output requiring human review")
                with st.expander("Model input row (exact features sent to the pipeline)"):
                    st.dataframe(row, use_container_width=True, hide_index=True)
                log = st.session_state.setdefault("pred_log", [])
                log.append({"Time": res["timestamp"], **row.iloc[0].to_dict(),
                            "Prediction": res["label_text"],
                            "Probability": None if res["probability"] is None else round(res["probability"], 3),
                            "Tier": res["tier"]})
        if st.session_state.get("pred_log"):
            st.markdown("#### Predictions made this session")
            st.dataframe(pd.DataFrame(st.session_state["pred_log"]), use_container_width=True, hide_index=True)
        with st.expander("Limitations"):
            for item in pred.LIMITATIONS:
                st.markdown(f"- {item}")

    with tab2:
        m = meta["metrics"]
        mid = pred.MIDTERM_REPORTED
        table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (Breach=1)", "Recall (Breach=1)", "F1 (Breach=1)"],
            "Deployed model (this app)": [m["accuracy"], m["precision"], m["recall"], m["f1"]],
            "Midterm report": [mid["Accuracy"], mid["Precision (Breach=1)"], mid["Recall (Breach=1)"],
                               mid["F1 (Breach=1)"]],
        }).round(3)
        st.dataframe(table, use_container_width=True, hide_index=True)
        cm = m["confusion_matrix"]
        st.markdown(f"**Confusion matrix (deployed):** TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}  |  "
                    f"**Midterm:** TN=17, FP=15, FN=6, TP=28")
        st.caption(f"Majority-class baseline accuracy: {m['majority_baseline_accuracy']:.3f}. "
                   f"Trained on {meta['n_train']} / tested on {meta['n_test']} labeled records "
                   f"(80/20 split, random_state=42). Trained {meta['trained_at']} with scikit-learn "
                   f"{meta['sklearn_version']}.")
        imp = pd.Series(meta["feature_importances"]).reset_index()
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Feature importances (aggregated)").update_yaxes(autorange="reversed"),
                        use_container_width=True)
        st.info("If the deployed numbers differ from the midterm, replace models/trained_model.joblib with the "
                "model exported from your midterm notebook (see README) and explain any remaining differences "
                "in the deployment report.")

    with tab3:
        st.markdown("Score every request in the active dataset (any status) with the saved pipeline.")
        if st.button("Score dataset"):
            scored = pred.score_dataframe(model, active_df)
            cols = ["Request_ID", "Service_Type", "Assigned_Office", "Status", "Breach_Probability",
                    "Risk_Tier", "Predicted_Breach_Flag"]
            st.dataframe(scored[cols], use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                               "scored_requests.csv", "text/csv")


def page_presentation():
    st.header("Presentation Guide")
    st.caption("A ready-to-present walkthrough of the system, with cues for what to show on screen.")
    st.info("Suggested duration: 5–7 minutes. Use the sidebar to move to each page when you reach its demo cue.")

    meta = None
    try:
        meta = load_meta()
    except Exception:
        st.warning("Model metadata is unavailable. Skip the model version and accuracy details below.")

    accuracy = f"{meta['metrics']['accuracy']:.1%}" if meta else "unavailable"
    model_version = meta["model_version"] if meta else "unavailable"
    sections = [
        (
            "1. Opening · 30 seconds",
            "Show: Home",
            "Good morning. Today, I’m presenting the Student Service Request Monitoring System, "
            "a decision-support application for monitoring student service requests and identifying "
            "possible SLA breaches. An SLA, or service-level agreement, is the target time for completing "
            "a request. The system estimates whether a request may exceed that target. It is designed to "
            "help staff identify requests that may need attention—not to make decisions on their behalf. "
            "The application currently uses synthetic sample data, not real student records.",
        ),
        (
            "2. Home · 45 seconds",
            "Show: Home page metrics and overview",
            f"The Home page gives a quick overview of the system. It currently shows {len(active_df):,} "
            f"loaded requests, model version {model_version}, and a deployed test accuracy of {accuracy}. "
            "That accuracy describes performance on the model’s test data; it does not guarantee that an "
            "individual prediction will be correct. The page also summarizes the system’s focus: making "
            "backlogs visible, highlighting possible risks, and keeping final decisions with staff.",
        ),
        (
            "3. Dataset · 45 seconds",
            "Show: Dataset",
            "On the Dataset page, staff can review the bundled sample or upload a CSV file. The system "
            "checks for required columns and shows a preview and data-quality summary, including missing "
            "values and duplicate records. This helps identify input-data problems before analysis or prediction.",
        ),
        (
            "4. Data Analysis · 45 seconds",
            "Show: Data Analysis",
            "The Data Analysis page explores request volume and patterns, including differences by service "
            "type or office and how breach rates vary. These charts can help staff notice potential "
            "bottlenecks. They describe the available data; they do not prove why a delay occurred.",
        ),
        (
            "5. Prediction · 1 minute",
            "Show: Prediction, then the individual prediction form",
            "The Prediction page supports individual and batch scoring. For an individual request, a user "
            "enters submission-time information and reviews the model’s estimate and related details. "
            "The system avoids using outcome fields such as actual processing time or final status for "
            "submission-time inference. A prediction is a signal for staff review—not a decision or a promise.",
        ),
        (
            "6. Dashboard · 1 minute",
            "Show: Dashboard",
            "The Dashboard brings the operational view together. It presents summary indicators and a "
            "request register or risk queue to help staff review requests and workflow status. Where "
            "available, authorized staff can record a final decision. The model may highlight a possible "
            "risk, but staff remain responsible for deciding what action to take.",
        ),
        (
            "7. System Design · 45 seconds",
            "Show: System Design",
            "The System Design page presents the system’s requirements, actors, use-case diagram, and "
            "workflow. It connects the application’s features to the needs of student services staff "
            "and administrators.",
        ),
        (
            "8. Closing · 30 seconds",
            "Return to: Home",
            "To summarize, this system combines data validation, analysis, SLA-risk prediction, and a "
            "staff-facing dashboard. Its purpose is to help staff see patterns and prioritize requests "
            "for review. Predictions have limitations, and the current data is synthetic, so the system "
            "should be evaluated with properly authorized real-world data before operational deployment. "
            "Thank you. I’m ready for your questions.",
        ),
    ]

    for title, cue, script in sections:
        with st.expander(title, expanded=title.startswith("1.")):
            st.markdown(f"**{cue}**")
            st.write(script)

    st.caption("Reminder: describe model predictions as decision-support estimates, not guaranteed outcomes.")


# ----------------------------------------------------------------- router
if menu == "Home":
    page_home()
elif menu == "Dataset":
    page_dataset()
elif menu == "Data Analysis":
    eda.render(active_df)
elif menu == "Prediction":
    page_prediction()
elif menu == "Dashboard":
    _model, _meta = model_or_stop()
    dash.render_dashboard(active_df, _model, _meta, role)
elif menu == "System Design":
    try:
        _meta = load_meta()
    except Exception:
        _meta = {}
    dash.render_system_design(_meta)
elif menu == "Presentation":
    page_presentation()
