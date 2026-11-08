"""Module 4 (decision-support dashboard) and Module 5 (SAD integration page)."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import prediction as pred
from modules.data_processing import labeled
from modules.exploratory_analysis import breach_rate_table

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ROLES = ["Service Officer", "Administrator"]
DECISIONS = [
    "Confirm risk - escalate / fast-track",
    "Override - keep in standard queue",
    "Request more information from requester",
]


def render_dashboard(df: pd.DataFrame, model, meta: dict, role: str):
    st.header("Decision-Support Dashboard")
    if df is None or df.empty:
        st.warning("No data available. Load a valid dataset on the Dataset page first.")
        return

    lab = labeled(df)
    open_df = df[df["Status"].isin(["Open", "Processing"])].copy()
    scored = pred.score_dataframe(model, open_df)
    flagged = scored[scored["Predicted_Breach_Flag"] == 1] if not scored.empty else scored
    delayed_requests = int((lab["SLA_Breached"] == 1).sum()) if not lab.empty else 0
    pending_requests = int((df["Status"] == "Open").sum())
    processing_requests = int((df["Status"] == "Processing").sum())
    completed_requests = int(((df["Status"] == "Completed") & (df["SLA_Breached"] == 0)).sum())
    breach_rate = (lab["SLA_Breached"].mean() * 100) if not lab.empty else 0.0

    st.markdown('<div class="section-title">Operations Snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k4, k5 = st.columns(2)
    for col, label, value, meta_text in [
        (k1, "Pending requests", f"{pending_requests:,}", "Waiting to be processed"),
        (k2, "Processing", f"{processing_requests:,}", "Currently being handled"),
        (k3, "Completed on time", f"{completed_requests:,}", "Met the service target"),
        (k4, "Delayed requests", f"{delayed_requests:,}", "Completed beyond the SLA"),
        (k5, "SLA breach rate", f"{breach_rate:.1f}%", "Of completed requests"),
    ]:
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-meta">{meta_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(f"Model v{meta.get('model_version', '?')} (trained {meta.get('trained_at', '?')}). "
               "Flags are advisory and require human review.")

    st.markdown('<div class="section-title">Priority Monitoring</div>', unsafe_allow_html=True)
    if len(flagged) > 0:
        st.markdown(
            f"<div class=\"alert-box\">{len(flagged)} outstanding requests are flagged for staff review. Predictions are advisory and must not be used to deny service.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class=\"alert-box\">No outstanding requests are currently flagged as high risk. Continue standard workflow monitoring.</div>",
            unsafe_allow_html=True,
        )

    # --- workload / delay charts
    c1, c2 = st.columns(2)
    off = breach_rate_table(df, "Assigned_Office")
    if not off.empty:
        c1.plotly_chart(px.bar(off, x="Assigned_Office", y="Breach_Rate_%", color="Breach_Rate_%",
                               range_y=[0, 100], title="SLA breach rate by office (completed requests)"),
                        use_container_width=True)
    if len(open_df):
        work = open_df.groupby("Assigned_Office").agg(Outstanding_Requests=("Request_ID", "count"),
                                                      Avg_Queue=("Queue_Length", "mean")).reset_index()
        c2.plotly_chart(px.bar(work, x="Assigned_Office", y="Outstanding_Requests", color="Avg_Queue",
                               title="Outstanding workload by office (colour = average queue length)"),
                        use_container_width=True)
    else:
        c2.info("No pending or in-progress requests in this dataset.")

    # --- review queue
    st.subheader("Risk / review queue")
    st.caption("Open requests are Pending; records marked Processing are in progress. Select a row for staff review.")
    if scored.empty:
        st.info("There are no open requests to score.")
        queue = scored
    else:
        filter_col1, filter_col2, filter_col3 = st.columns([1.35, 1, 1])
        search = filter_col1.text_input("Search requests", placeholder="Request ID, service, or office")
        offices = filter_col2.multiselect(
            "Office",
            sorted(scored["Assigned_Office"].dropna().unique()),
            default=sorted(scored["Assigned_Office"].dropna().unique()),
        )
        tiers = filter_col3.multiselect("Risk tier", ["High", "Elevated", "Low"], default=["High", "Elevated"])
        sort_by = st.selectbox(
            "Sort queue by",
            ["Highest breach probability", "Longest queue", "Shortest SLA"],
        )
        queue = scored[
            scored["Assigned_Office"].isin(offices) & scored["Risk_Tier"].isin(tiers)
        ].copy()
        if search.strip():
            search_text = search.strip()
            search_mask = (
                queue["Request_ID"].astype(str).str.contains(search_text, case=False, na=False)
                | queue["Service_Type"].astype(str).str.contains(search_text, case=False, na=False)
                | queue["Assigned_Office"].astype(str).str.contains(search_text, case=False, na=False)
            )
            queue = queue[search_mask]
        sort_column = {
            "Highest breach probability": "Breach_Probability",
            "Longest queue": "Queue_Length",
            "Shortest SLA": "SLA_Days",
        }[sort_by]
        queue = queue.sort_values(sort_column, ascending=(sort_by == "Shortest SLA"))
        queue["Status"] = queue["Status"].map({"Open": "🔵 Pending", "Processing": "🟦 Processing"})
        queue["Risk"] = queue["Risk_Tier"].map(
            {"High": "🔴 High", "Elevated": "🟠 Elevated", "Low": "🟢 Low"}
        )
        cols = ["Request_ID", "Status", "Service_Type", "Assigned_Office", "Priority", "Queue_Length",
                "Queue_Category", "SLA_Days", "Required_Documents_Complete", "Breach_Probability", "Risk",
                "Recommended_Action"]
        if queue.empty:
            st.info("No requests match the current search and filters.")
        else:
            st.dataframe(
                queue[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn("Status"),
                    "Risk": st.column_config.TextColumn("Risk tier"),
                    "Breach_Probability": st.column_config.ProgressColumn(
                        "Breach probability", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )
        st.caption(f"{len(queue)} request(s) match the current filters. Table columns can be sorted by clicking their headers.")

    with st.expander("Search and filter the full request register"):
        register = df.copy()
        register["Workflow_Status"] = "⚪ Cancelled"
        register.loc[register["Status"] == "Open", "Workflow_Status"] = "🔵 Pending"
        register.loc[register["Status"] == "Processing", "Workflow_Status"] = "🟦 Processing"
        completed_mask = register["Status"] == "Completed"
        register.loc[completed_mask & register["SLA_Breached"].eq(0), "Workflow_Status"] = "🟢 Completed"
        register.loc[completed_mask & register["SLA_Breached"].eq(1), "Workflow_Status"] = "🔴 Delayed"
        register_search = st.text_input("Search register", placeholder="Request ID, service, or office")
        status_options = sorted(register["Workflow_Status"].dropna().unique())
        selected_statuses = st.multiselect("Workflow status", status_options, default=status_options)
        register_view = register[register["Workflow_Status"].isin(selected_statuses)]
        if register_search.strip():
            query = register_search.strip()
            register_mask = (
                register_view["Request_ID"].astype(str).str.contains(query, case=False, na=False)
                | register_view["Service_Type"].astype(str).str.contains(query, case=False, na=False)
                | register_view["Assigned_Office"].astype(str).str.contains(query, case=False, na=False)
            )
            register_view = register_view[register_mask]
        register_cols = ["Request_ID", "Workflow_Status", "Service_Type", "Assigned_Office", "Priority",
                         "Queue_Length", "SLA_Days", "Processing_Days", "SLA_Breached"]
        st.dataframe(register_view[register_cols], use_container_width=True, hide_index=True)
        st.caption(f"{len(register_view):,} request(s) shown. Sort any column by selecting its header.")

    # --- interpretation
    st.subheader("Interpretation")
    bullets = []
    if not off.empty:
        top = off.iloc[0]
        bullets.append(f"**{top['Assigned_Office']}** has the highest historical breach rate "
                       f"({top['Breach_Rate_%']}%). Check its staffing and approval steps first.")
    if len(flagged):
        no_docs = int((flagged["Required_Documents_Complete"] == "No").sum())
        high_q = int((flagged["Queue_Category"] == "High").sum())
        bullets.append(f"{len(flagged)} open request(s) are flagged: {no_docs} lack complete documents and "
                       f"{high_q} sit in a high queue (>15).")
    bullets.append("A flag means *review*, not *reject*. The model produced many false alarms in testing.")
    st.markdown("\n".join(f"- {b}" for b in bullets))
    st.markdown("**Recommended human follow-up:** contact requesters with missing documents, rebalance workload "
                "from high-queue offices, and fast-track high-risk requests approaching their SLA date.")

    # --- human decision recording
    st.subheader("Human review and final decision")
    if "decision_log" not in st.session_state:
        st.session_state["decision_log"] = []
    if role == "Service Officer" and not queue.empty:
        with st.form("decision_form", clear_on_submit=True):
            rid = st.selectbox("Request", queue["Request_ID"].tolist())
            decision = st.radio("Final decision", DECISIONS)
            reviewer = st.text_input("Reviewer name or ID")
            note = st.text_area("Justification (required)")
            submitted = st.form_submit_button("Record decision")
        if submitted:
            if not reviewer.strip() or not note.strip():
                st.error("Reviewer and justification are required before a decision can be recorded.")
            else:
                row = queue[queue["Request_ID"] == rid].iloc[0]
                st.session_state["decision_log"].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Request_ID": rid, "Decision": decision, "Reviewer": reviewer.strip(),
                    "Note": note.strip(), "Model_Probability": round(float(row["Breach_Probability"]), 3),
                    "Model_Version": meta.get("model_version", "?"), "Role": role})
                st.success(f"Decision recorded for {rid}.")
    elif role == "Administrator":
        st.info("Administrators can view results and the decision log but cannot record final decisions. "
                "Switch the role in the sidebar to 'Service Officer' to record one.")
    log = pd.DataFrame(st.session_state["decision_log"])
    if log.empty:
        st.caption("No decisions recorded in this session. (The log is session-only in this demo; "
                   "a production system would store it in a database.)")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)
        st.download_button("Download decision log (CSV)", log.to_csv(index=False).encode("utf-8"),
                           "decision_log.csv", "text/csv")


def render_system_design(meta: dict):
    st.header("System Design (SAD Integration)")
    st.markdown("This page documents the analysis artifacts that correspond to the implemented features. "
                "Findings cited as *midterm* come from the submitted Laboratory Examination Report.")

    st.subheader("Functional requirements (evidence-based)")
    fr = pd.DataFrame([
        ["FR-1", "Display SLA breach performance grouped by assigned office.",
         "Midterm: Scholarship Office 78.9% (30/38), Finance 75.6%.", "Data Analysis, Dashboard"],
        ["FR-2", "Categorize queue status (Low/Medium/High) and highlight high-queue requests.",
         "Midterm: elevated breach risk in high backlogs.", "Data Analysis, Dashboard"],
        ["FR-3", "Generate an advisory delay-risk flag and probability from submission-time features "
                 "using the saved Decision Tree pipeline.",
         "Midterm: accuracy 68.2% vs 51.5% baseline, recall 0.824.", "Prediction"],
        ["FR-4", "Show model metrics, top feature drivers and limitations with every prediction.",
         "Midterm top drivers: SLA_Days, Service_Type, Required_Documents_Complete.", "Prediction"],
        ["FR-5", "Route flagged requests to a human review queue; only a Service Officer may record the "
                 "final decision with justification and timestamp.",
         "Midterm: FP rate 46.8%, so automated denial would be unjust.", "Dashboard"],
    ], columns=["ID", "Requirement", "Evidence", "Implemented in"])
    st.dataframe(fr, use_container_width=True, hide_index=True)

    st.subheader("Non-functional requirements")
    nfr = pd.DataFrame([
        ["NFR-1 Data validation", "Uploads are checked for schema, empty files, types and dates with clear error "
                                  "messages before use.", "Dataset"],
        ["NFR-2 Performance", "The model is loaded once (cached) and never retrained on refresh; pages respond "
                              "within a few seconds for datasets up to a few thousand rows.", "All"],
        ["NFR-3 Privacy and transparency", "Only synthetic or authorized data is deployed; no credentials are "
                                           "committed; model version, timestamp and limitations are visible.", "Prediction, Dashboard"],
    ], columns=["Requirement", "Description", "Where"])
    st.dataframe(nfr, use_container_width=True, hide_index=True)

    st.subheader("Actors and permissions")
    perms = pd.DataFrame([
        ["Administrator", "Yes", "Yes", "Yes (read-only)", "No"],
        ["Service Officer", "Yes", "Yes", "Yes", "Yes"],
    ], columns=["Actor", "Upload / validate data", "View predictions & dashboard", "View decision log",
                "Record final decision"])
    st.dataframe(perms, use_container_width=True, hide_index=True)
    st.caption("Request submission by students happens in the upstream service-request system and is outside "
               "this application's boundary. Role switching in the sidebar is a demo of authorization rules; "
               "production use needs real authentication.")

    st.subheader("Updated use case diagram")
    img = ASSETS / "use_case_diagram.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.error("assets/use_case_diagram.png is missing. Run scripts/make_use_case_diagram.py.")

    st.subheader("Analytical process flow / decision workflow")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eaf1fb", fontname="Helvetica"];
      A [label="Upload CSV"]; B [label="Validate schema\\n+ clean data"];
      C [label="Engineer features\\n(Queue_Category, Weekday)"];
      D [label="Saved pipeline\\npredicts breach risk"];
      E [label="Risk tier +\\nrecommended action"];
      F [label="Service Officer\\nreviews flag", fillcolor="#fdecec"];
      G [label="Record final decision\\n(with justification)", fillcolor="#fdecec"];
      H [label="Follow-up: reassign,\\nfast-track, request docs"];
      A -> B -> C -> D -> E -> F -> G -> H;
      B -> A [style=dashed, label="invalid", fontsize=9];
    }
    """)
    st.caption("No step in this workflow automatically rejects or denies a request.")
