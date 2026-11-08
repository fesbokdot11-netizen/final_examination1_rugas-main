"""Module 2 - Exploratory Data Analysis (descriptive stats, class distribution, interactive charts)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import QUEUE_ORDER, WEEKDAYS, labeled

GROUP_OPTIONS = ["Service_Type", "Assigned_Office", "Requester_Type", "Priority",
                 "Queue_Category", "Submission_Weekday", "Required_Documents_Complete"]


def _order(col):
    if col == "Queue_Category":
        return {col: QUEUE_ORDER}
    if col == "Submission_Weekday":
        return {col: WEEKDAYS}
    return {}


def breach_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    lab = labeled(df)
    if lab.empty:
        return pd.DataFrame(columns=[col, "Completed", "Breaches", "Breach_Rate_%"])
    g = lab.groupby(col)["SLA_Breached"].agg(Completed="count", Breaches="sum").reset_index()
    g["Breach_Rate_%"] = (g["Breaches"] / g["Completed"] * 100).round(1)
    return g.sort_values("Breach_Rate_%", ascending=False)


def render(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    if df is None or df.empty:
        st.warning("No data available. Load a valid dataset on the Dataset page first.")
        return

    # interactive filters
    with st.expander("Filters", expanded=False):
        req = st.multiselect("Requester type", sorted(df["Requester_Type"].dropna().unique()),
                             default=sorted(df["Requester_Type"].dropna().unique()))
        off = st.multiselect("Assigned office", sorted(df["Assigned_Office"].dropna().unique()),
                             default=sorted(df["Assigned_Office"].dropna().unique()))
    view = df[df["Requester_Type"].isin(req) & df["Assigned_Office"].isin(off)]
    if view.empty:
        st.warning("The current filters return no rows. Widen the filters.")
        return
    lab = labeled(view)
    st.caption(f"{len(view):,} requests in view, {len(lab):,} completed with a known outcome.")

    st.subheader("1. Descriptive statistics")
    num_cols = [c for c in ["Queue_Length", "SLA_Days", "Processing_Days", "Satisfaction_Score"] if c in view.columns]
    c1, c2 = st.columns(2)
    c1.markdown("**Numeric fields**")
    c1.dataframe(view[num_cols].describe().T.round(2), use_container_width=True)
    c2.markdown("**Categorical fields**")
    cat_cols = ["Service_Type", "Requester_Type", "Assigned_Office", "Priority", "Status"]
    c2.dataframe(view[cat_cols].describe().T, use_container_width=True)

    st.subheader("2. Class distribution (target: SLA_Breached)")
    if lab.empty:
        st.info("No completed requests with known outcomes in this view, so the target cannot be shown.")
    else:
        dist = lab["SLA_Breached"].map({1: "Breached (1)", 0: "Within SLA (0)"}).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        dist["Share_%"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
        a, b = st.columns([1, 1])
        a.dataframe(dist, use_container_width=True, hide_index=True)
        fig = px.pie(dist, names="Class", values="Count", hole=0.45, title="SLA breach class balance")
        b.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Interactive charts")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Request volume", "Breach rate", "Processing vs SLA", "Queue vs breach", "Weekday x office"])

    with tab1:
        col = st.selectbox("Group volume by", GROUP_OPTIONS, key="eda_vol")
        vol = view.groupby(col).size().reset_index(name="Requests")
        fig = px.bar(vol.sort_values("Requests", ascending=False), x=col, y="Requests",
                     color="Requests", title=f"Request volume by {col}", category_orders=_order(col))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col = st.selectbox("Breach rate by", GROUP_OPTIONS, index=1, key="eda_rate")
        tbl = breach_rate_table(view, col)
        if tbl.empty:
            st.info("No labeled requests to compute breach rates.")
        else:
            fig = px.bar(tbl, x=col, y="Breach_Rate_%", color="Breach_Rate_%",
                         hover_data=["Completed", "Breaches"], range_y=[0, 100],
                         title=f"SLA breach rate (%) by {col}", category_orders=_order(col))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    with tab3:
        if lab.empty:
            st.info("No labeled requests to plot.")
        else:
            plot = lab.copy()
            plot["Outcome"] = plot["SLA_Breached"].map({1: "Breached", 0: "Within SLA"})
            fig = px.histogram(plot, x="Processing_Days", color="Outcome", barmode="overlay",
                               nbins=20, opacity=0.7, hover_data=["Service_Type", "Assigned_Office"],
                               title="Processing days by outcome")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Processing_Days is an OUTCOME field. It is used for analysis only and never as a model input.")

    with tab4:
        if lab.empty:
            st.info("No labeled requests to plot.")
        else:
            plot = lab.copy()
            plot["Outcome"] = plot["SLA_Breached"].map({1: "Breached", 0: "Within SLA"})
            fig = px.box(plot, x="Outcome", y="Queue_Length", color="Outcome", points="all",
                         hover_data=["Request_ID", "Service_Type"],
                         title="Queue length at submission vs outcome")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if lab.empty:
            st.info("No labeled requests to plot.")
        else:
            pivot = lab.pivot_table(index="Assigned_Office", columns="Submission_Weekday",
                                    values="SLA_Breached", aggfunc="mean")
            pivot = pivot.reindex(columns=[d for d in WEEKDAYS if d in pivot.columns]) * 100
            fig = px.imshow(pivot.round(1), text_auto=True, aspect="auto", color_continuous_scale="Reds",
                            labels=dict(color="Breach %"), title="Breach rate (%) by office and submission weekday")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("4. Findings")
    office = breach_rate_table(view, "Assigned_Office")
    vol = view.groupby("Service_Type").size().sort_values(ascending=False)
    if not office.empty:
        top = office.iloc[0]
        st.markdown(
            f"- **{vol.index[0]}** has the highest request volume ({int(vol.iloc[0])} requests).\n"
            f"- **{top['Assigned_Office']}** has the highest breach rate in this view "
            f"({top['Breach_Rate_%']}%, {int(top['Breaches'])} of {int(top['Completed'])} completed requests).\n"
            "- Breach analysis uses completed requests only; open and cancelled requests have no known outcome.")
