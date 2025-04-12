import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv("results.csv")

# Streamlit app setup
st.title("Machine Learning Model Performance Dashboard")
st.write("Compare performance metrics across different models.")

# Sidebar for model selection
st.sidebar.header("Filters")
selected_models = st.sidebar.multiselect(
    "Select Models", df["Model"].unique(), default=df["Model"].unique()
)
filtered_df = df[df["Model"].isin(selected_models)]

# Section 1: Raw Data Table
st.header("Model Results")
st.dataframe(filtered_df)

# Section 2: Bar Chart for Metrics
st.header("Performance Comparison")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
fig_bar = px.bar(
    filtered_df,
    x="Model",
    y=metrics,
    barmode="group",
    title="Metric Comparison Across Models",
    labels={"value": "Score", "Model": "Model"}
)
st.plotly_chart(fig_bar)

# Section 3: Radar Chart
st.header("Radar Chart")
fig_radar = go.Figure()
for _, row in filtered_df.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row[m] for m in metrics],
        theta=metrics,
        fill="toself",
        name=row["Model"]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0])), showlegend=True)
st.plotly_chart(fig_radar)

# Section 4: Download Option
st.header("Export Data")
csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "filtered_model_results.csv", "text/csv")