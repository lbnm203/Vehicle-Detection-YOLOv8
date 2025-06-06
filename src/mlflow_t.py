import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.express as px


def display_mlflow_runs():
    st.header("MLflow Experiment Tracking")

    # Set up MLflow tracking
    mlflow_tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Get all experiments
    experiments = client.search_experiments()
    if not experiments:
        st.info("No MLflow experiments found. Train a model first.")
        return

    # Only show the "Default" experiment (usually named "Default" or id=0)
    # Find the default experiment (by name or id)
    default_experiment = None
    for exp in experiments:
        if exp.name.lower() == "default" or exp.experiment_id == "0":
            default_experiment = exp
            break

    if not default_experiment:
        st.info("No 'Default' MLflow experiment found.")
        return

    experiment_names = [default_experiment.name]
    experiment_ids = [default_experiment.experiment_id]
    default_exp_idx = 0
    selected_exp_idx = st.selectbox(
        "Select Experiment",
        range(len(experiment_names)),
        format_func=lambda i: experiment_names[i],
        index=default_exp_idx
    )
    selected_exp_id = experiment_ids[selected_exp_idx]

    # Get runs for the selected experiment
    runs = client.search_runs(experiment_ids=[selected_exp_id])
    if not runs:
        st.info(
            f"No runs found for experiment '{experiment_names[selected_exp_idx]}'")
        return

    # Create a dataframe with run information
    run_data = []
    for run in runs:
        run_info = {
            "Run ID": run.info.run_id,
            "Run Name": run.data.tags.get("mlflow.runName", "Unnamed"),
            "Status": run.info.status,
            "Start Time": pd.to_datetime(run.info.start_time, unit='ms'),
            "End Time": pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None,
        }

        # Add parameters
        for key, value in run.data.params.items():
            run_info[f"param.{key}"] = value

        # Add metrics (last recorded value)
        for key, value in run.data.metrics.items():
            run_info[f"metric.{key}"] = value

        run_data.append(run_info)

    runs_df = pd.DataFrame(run_data)

    # Display runs table
    st.subheader("Runs")
    st.dataframe(runs_df)

    # Select a run to view details
    selected_run_id = st.selectbox(
        "Select a run to view details",
        runs_df["Run ID"].tolist(),
        format_func=lambda run_id: f"{run_id} ({runs_df[runs_df['Run ID'] == run_id]['Run Name'].iloc[0]})"
    )

    # Get the selected run
    selected_run = client.get_run(selected_run_id)

    # Display run details
    st.subheader("Run Details")

    # Parameters
    st.write("#### Parameters")
    params_df = pd.DataFrame(list(selected_run.data.params.items()), columns=[
                             "Parameter", "Value"])
    st.table(params_df)

    # Metrics
    st.write("#### Metrics")
    # Remove this problematic line:
    # metrics = client.get_metric_history(selected_run_id, "")

    # Group metrics by name
    metric_names = set()
    # Get all metrics for the run
    metrics = selected_run.data.metrics
    for metric_name in metrics.keys():
        metric_names.add(metric_name)

    # Create plots for each metric
    if metric_names:
        tabs = st.tabs(list(metric_names))

        for i, metric_name in enumerate(metric_names):
            with tabs[i]:
                metric_history = client.get_metric_history(
                    selected_run_id, metric_name)

                if metric_history:
                    # Create dataframe for the metric
                    metric_df = pd.DataFrame([
                        {"step": m.step, "value": m.value,
                            "timestamp": pd.to_datetime(m.timestamp, unit='ms')}
                        for m in metric_history
                    ])

                    # Create plot
                    fig = px.line(
                        metric_df,
                        x="step",
                        y="value",
                        title=f"{metric_name} over steps",
                        labels={"step": "Step", "value": "Value"}
                    )
                    st.plotly_chart(fig)

                    # Show data table
                    st.dataframe(metric_df)
    else:
        st.info("No metrics found for this run")
