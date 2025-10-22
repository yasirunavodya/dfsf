# ui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from inference import RFPowerForecaster

# Configure page
st.set_page_config(page_title="Solar RF Forecast Dashboard", layout="wide")
st.title("🌞 Solar Power Forecasting Dashboard")


# Initialize forecaster
@st.cache_resource
def load_forecaster():
    return RFPowerForecaster()


forecaster = load_forecaster()

# Initialize session state
if "hist_df" not in st.session_state:
    st.session_state.hist_df = None
if "fut_df" not in st.session_state:
    st.session_state.fut_df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📁 Dataset Upload",
        "📊 Input Data Visualization",
        "🔮 Forecasting",
        "📈 Forecast Visualization",
    ]
)

with tab1:
    st.header("Dataset Upload")
    st.markdown("Upload your datasets to get started with solar power forecasting.")

    col1, col2 = st.columns(2)

    # with col1:
    st.subheader("Historical Data")
    hist_file = st.file_uploader(
        "Upload recent history CSV (must include 'Date' + weather columns)",
        type=["csv"],
        key="hist_upload",
    )

    if hist_file is not None:
        try:
            st.session_state.hist_df = pd.read_csv(hist_file)
            st.success("✅ Historical data uploaded successfully!")
            st.info(
                f"Shape: {st.session_state.hist_df.shape[0]} rows × {st.session_state.hist_df.shape[1]} columns"
            )

            # Show preview
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.hist_df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")

    # with col2:
    #     st.subheader("Future Weather Data (Optional)")
    #     fut_file = st.file_uploader(
    #         "Upload future weather CSV (3 rows for next 3 hours)",
    #         type=["csv"],
    #         key="fut_upload",
    #     )

    #     if fut_file is not None:
    #         try:
    #             st.session_state.fut_df = pd.read_csv(fut_file)
    #             st.success("✅ Future weather data uploaded successfully!")
    #             st.info(
    #                 f"Shape: {st.session_state.fut_df.shape[0]} rows × {st.session_state.fut_df.shape[1]} columns"
    #             )

    #             # Show preview
    #             st.subheader("Future Weather Preview")
    #             st.dataframe(st.session_state.fut_df, use_container_width=True)

    #         except Exception as e:
    #             st.error(f"Error loading future weather data: {str(e)}")

with tab2:
    st.header("Input Data Visualization")

    if st.session_state.hist_df is not None:
        df = st.session_state.hist_df.copy()

        # Dataset information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                date_range = df["Date"].max() - df["Date"].min()
                st.metric("Time Range", f"{date_range.days} days")

        # Data types and missing values
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Types:**")
            data_types = df.dtypes.to_frame("Data Type")
            data_types["Column"] = data_types.index
            st.dataframe(data_types[["Column", "Data Type"]], use_container_width=True)

        with col2:
            st.write("**Missing Values:**")
            missing = df.isnull().sum().to_frame("Missing Count")
            missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
            missing["Column"] = missing.index
            missing_display = missing[missing["Missing Count"] > 0][
                ["Column", "Missing Count", "Missing %"]
            ]
            if len(missing_display) > 0:
                st.dataframe(missing_display, use_container_width=True)
            else:
                st.success("No missing values found!")

        # Statistical summary
        #st.subheader("Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        #if len(numeric_cols) > 0:
            #st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Time series plots
        if "Date" in df.columns:
            st.subheader("Time Series Visualization")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")

            # Select columns to plot
            plot_cols = st.multiselect(
                "Select columns to visualize:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:3]
                if len(numeric_cols) >= 3
                else numeric_cols.tolist(),
            )

            if plot_cols:
                fig = make_subplots(
                    rows=len(plot_cols),
                    cols=1,
                    subplot_titles=plot_cols,
                    shared_xaxes=True,
                )

                for i, col in enumerate(plot_cols):
                    fig.add_trace(
                        go.Scatter(x=df["Date"], y=df[col], name=col, mode="lines"),
                        row=i + 1,
                        col=1,
                    )

                fig.update_layout(height=200 * len(plot_cols), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Feature Correlation Heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Please upload historical data in the Dataset Upload tab to see visualizations."
        )

with tab3:
    st.header("Solar Power Forecasting")

    if st.session_state.hist_df is not None:
        st.subheader("Forecast Configuration")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.info("Historical data is loaded and ready for forecasting.")
        with col2:
            forecast_button = st.button(
                "🔮 Run Forecast", type="primary", use_container_width=True
            )

        if forecast_button:
            try:
                with st.spinner("Running forecast model..."):
                    predictions = forecaster.forecast_next3(
                        history_df=st.session_state.hist_df,
                        future_weather_df=st.session_state.fut_df,
                    )
                    st.session_state.predictions = predictions

                st.success("✅ Forecast completed successfully!")

                # Display results
                st.subheader("Forecast Results")
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Format predictions for display
                    display_preds = predictions.copy()
                    display_preds.index = display_preds.index.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    display_preds["Predicted_Power_Output"] = display_preds[
                        "Predicted_Power_Output"
                    ].round(4)

                    st.dataframe(display_preds, use_container_width=True)

                with col2:
                    # Summary statistics
                    avg_power = predictions["Predicted_Power_Output"].mean()
                    max_power = predictions["Predicted_Power_Output"].max()
                    min_power = predictions["Predicted_Power_Output"].min()

                    #st.metric("Average Power", f"{avg_power:.2f}")
                    #st.metric("Maximum Power", f"{max_power:.2f}")
                    #st.metric("Minimum Power", f"{min_power:.2f}")

                # Quick visualization
                st.subheader("Quick Forecast Preview")
                fig = px.line(
                    x=predictions.index,
                    y=predictions["Predicted_Power_Output"],
                    title="Predicted Power Output - Next 3 Hours",
                    labels={"x": "Time", "y": "Power Output"},
                )
                fig.update_traces(mode="lines+markers")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
                st.error("Please ensure your data has the required columns and format.")
    else:
        st.warning(
            "Please upload historical data in the Dataset Upload tab before running forecasts."
        )

with tab4:
    st.header("Forecast Visualization")

    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions

        # Combined historical and forecast plot
        if st.session_state.hist_df is not None:
            st.subheader("Historical vs Forecast Data")

            hist_df = st.session_state.hist_df.copy()
            if "Date" in hist_df.columns:
                hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors="coerce")
                hist_df = hist_df.sort_values("Date").tail(24)  # Last 24 hours

                # Check if target column exists in historical data
                target_col = None
                possible_targets = [
                    "Power Output",
                    "power_output",
                    "target",
                    "Power_Output",
                ]
                for col in possible_targets:
                    if col in hist_df.columns:
                        target_col = col
                        break

                fig = go.Figure()

                # Add historical data if target column exists
                if target_col:
                    fig.add_trace(
                        go.Scatter(
                            x=hist_df["Date"],
                            y=hist_df[target_col],
                            mode="lines+markers",
                            name="Historical Power Output",
                            line=dict(color="blue"),
                        )
                    )

                # Add forecast data
                fig.add_trace(
                    go.Scatter(
                        x=predictions.index,
                        y=predictions["Predicted_Power_Output"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="red", dash="dash"),
                        marker=dict(size=8),
                    )
                )

                fig.update_layout(
                    title="Solar Power: Historical Data vs Forecast",
                    xaxis_title="Time",
                    yaxis_title="Power Output",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

        # Detailed forecast analysis
        st.subheader("Forecast Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of predictions
            fig_bar = px.bar(
                x=predictions.index,
                y=predictions["Predicted_Power_Output"],
                title="Hourly Forecast Breakdown",
            )
            fig_bar.update_xaxes(title="Time")
            fig_bar.update_yaxes(title="Power Output")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Gauge chart for average forecast
            avg_forecast = predictions["Predicted_Power_Output"].mean()
            max_possible = (
                predictions["Predicted_Power_Output"].max() * 1.2
            )  # Assume 20% above max as upper limit

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_forecast,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Average Forecast"},
                    gauge={
                        "axis": {"range": [None, max_possible]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, max_possible * 0.3], "color": "lightgray"},
                            {
                                "range": [max_possible * 0.3, max_possible * 0.7],
                                "color": "gray",
                            },
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": max_possible * 0.9,
                        },
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Forecast statistics table
        st.subheader("Forecast Statistics")

        stats_data = {
            "Metric": ["Mean", "Median", "Standard Deviation", "Min", "Max", "Range"],
            "Value": [
                predictions["Predicted_Power_Output"].mean(),
                predictions["Predicted_Power_Output"].median(),
                predictions["Predicted_Power_Output"].std(),
                predictions["Predicted_Power_Output"].min(),
                predictions["Predicted_Power_Output"].max(),
                predictions["Predicted_Power_Output"].max()
                - predictions["Predicted_Power_Output"].min(),
            ],
        }

        stats_df = pd.DataFrame(stats_data)
        stats_df["Value"] = stats_df["Value"].round(4)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Download predictions
        st.subheader("Export Results")
        csv = predictions.to_csv()
        st.download_button(
            label="📥 Download Forecast Results as CSV",
            data=csv,
            file_name="solar_power_forecast.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Please run a forecast in the Forecasting tab to see visualizations here."
        )
