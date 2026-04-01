import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import io

# Set page config
st.set_page_config(page_title="Noisy Time Series Dashboard", layout="wide")

# Function to generate noisy time series data
def generate_time_series(n_points=200, trend_slope=0.1, seasonal_amplitude=5.0,
                        noise_std=2.0, seasonal_period=50):
    """
    Generate a noisy time series with trend, seasonality, and noise.

    Parameters:
    - n_points: Number of data points
    - trend_slope: Slope of linear trend
    - seasonal_amplitude: Amplitude of seasonal sine wave
    - noise_std: Standard deviation of noise
    - seasonal_period: Period of seasonal component

    Returns:
    - DataFrame with time, trend, seasonality, noise, and value columns
    """
    time = np.arange(n_points)

    # Linear trend
    trend = trend_slope * time

    # Seasonal component (sine wave)
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * time / seasonal_period)

    # Random noise
    noise = np.random.normal(0, noise_std, n_points)

    # Combined value
    value = trend + seasonality + noise

    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'trend': trend,
        'seasonality': seasonality,
        'noise': noise,
        'value': value
    })

    return df

# Function to detect anomalies using IQR method
def detect_anomalies(data, multiplier=1.5):
    """
    Detect anomalies using IQR method.

    Parameters:
    - data: Array-like data
    - multiplier: IQR multiplier for outlier detection

    Returns:
    - Boolean array indicating outliers
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (data < lower_bound) | (data > upper_bound)

# Function to create rolling mean prediction
def rolling_mean_prediction(data, window):
    """
    Create simple moving average prediction.

    Parameters:
    - data: Time series data
    - window: Window size for moving average

    Returns:
    - Predicted values
    """
    return data.rolling(window=window, center=True).mean()

# Main Streamlit app
def main():
    st.title("📊 Noisy Time Series Dashboard")

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    n_points = st.sidebar.slider("Number of points", 100, 1000, 200, 50)
    trend_slope = st.sidebar.slider("Trend slope", 0.0, 1.0, 0.1, 0.05)
    seasonal_amplitude = st.sidebar.slider("Seasonal amplitude", 0.0, 10.0, 5.0, 0.5)
    noise_std = st.sidebar.slider("Noise level", 0.0, 10.0, 2.0, 0.5)
    seasonal_period = st.sidebar.slider("Seasonal period", 10, 200, 50, 10)

    # Visualization controls
    st.sidebar.subheader("Visualization")
    rolling_window = st.sidebar.slider("Rolling window size", 5, 50, 20, 5)

    # Component visibility checkboxes
    st.sidebar.subheader("Show Components")
    show_trend = st.sidebar.checkbox("Trend", value=True)
    show_seasonality = st.sidebar.checkbox("Seasonality", value=True)
    show_noise = st.sidebar.checkbox("Noise", value=True)

    # Advanced features
    st.sidebar.subheader("Advanced Features")
    show_decomposition = st.sidebar.checkbox("Time Series Decomposition")
    show_anomalies = st.sidebar.checkbox("Anomaly Detection")
    show_prediction = st.sidebar.checkbox("Moving Average Prediction")

    # Bonus: File upload
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    # Generate or load data
    if uploaded_file is not None:
        # Load uploaded data
        df = pd.read_csv(uploaded_file)
        if 'value' not in df.columns:
            st.error("CSV must contain a 'value' column")
            return

        # Add time column if not present
        if 'time' not in df.columns:
            df['time'] = np.arange(len(df))

        # Generate synthetic components for uploaded data (approximation)
        df['trend'] = np.polyval(np.polyfit(df['time'], df['value'], 1), df['time'])
        df['seasonality'] = df['value'] - df['trend']
        df['noise'] = np.random.normal(0, df['value'].std() * 0.1, len(df))

        st.sidebar.success("Data loaded successfully!")
    else:
        # Generate synthetic data
        df = generate_time_series(n_points, trend_slope, seasonal_amplitude,
                                noise_std, seasonal_period)

    # Save dataset as CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.sidebar.download_button(
        label="📥 Download Dataset",
        data=csv_buffer.getvalue(),
        file_name="time_series_data.csv",
        mime="text/csv"
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Time Series Plot")

        # Create the main plot
        fig = go.Figure()

        # Add the combined value
        fig.add_trace(go.Scatter(x=df['time'], y=df['value'],
                               mode='lines', name='Value',
                               line=dict(color='blue', width=2)))

        # Add components if selected
        if show_trend:
            fig.add_trace(go.Scatter(x=df['time'], y=df['trend'],
                                   mode='lines', name='Trend',
                                   line=dict(color='red', dash='dash')))

        if show_seasonality:
            fig.add_trace(go.Scatter(x=df['time'], y=df['seasonality'],
                                   mode='lines', name='Seasonality',
                                   line=dict(color='green', dash='dot')))

        if show_noise:
            fig.add_trace(go.Scatter(x=df['time'], y=df['noise'],
                                   mode='lines', name='Noise',
                                   line=dict(color='orange', dash='dashdot')))

        # Add anomalies if selected
        if show_anomalies:
            anomalies = detect_anomalies(df['value'])
            anomaly_points = df[anomalies]
            fig.add_trace(go.Scatter(x=anomaly_points['time'], y=anomaly_points['value'],
                                   mode='markers', name='Anomalies',
                                   marker=dict(color='red', size=8, symbol='x')))

        # Add prediction if selected
        if show_prediction:
            prediction = rolling_mean_prediction(df['value'], rolling_window)
            fig.add_trace(go.Scatter(x=df['time'], y=prediction,
                                   mode='lines', name=f'MA Prediction (w={rolling_window})',
                                   line=dict(color='purple', width=3)))

        fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("📊 Summary Statistics")
        stats_df = df[['value']].describe().T
        st.dataframe(stats_df.style.format("{:.2f}"))

        # Additional plots
        st.subheader("Distribution")
        fig_hist = px.histogram(df, x='value', nbins=30, title="Histogram")
        st.plotly_chart(fig_hist, width='stretch')

        st.subheader("Box Plot")
        fig_box = px.box(df, y='value', title="Box Plot")
        st.plotly_chart(fig_box, width='stretch')

    # Rolling mean plot
    st.subheader("📉 Rolling Mean Analysis")
    rolling_mean = df['value'].rolling(window=rolling_window).mean()

    fig_rolling = make_subplots(rows=1, cols=2,
                               subplot_titles=("Original vs Rolling Mean", "Rolling Mean"))

    fig_rolling.add_trace(go.Scatter(x=df['time'], y=df['value'],
                                   mode='lines', name='Original'), row=1, col=1)
    fig_rolling.add_trace(go.Scatter(x=df['time'], y=rolling_mean,
                                   mode='lines', name='Rolling Mean',
                                   line=dict(color='red', width=3)), row=1, col=1)

    fig_rolling.add_trace(go.Scatter(x=df['time'], y=rolling_mean,
                                   mode='lines', name='Rolling Mean'), row=1, col=2)

    fig_rolling.update_layout(height=300)
    st.plotly_chart(fig_rolling, width='stretch')

    # Scatter plot
    st.subheader("🔸 Scatter Plot")
    fig_scatter = px.scatter(df, x='time', y='value', title="Time vs Value Scatter")
    st.plotly_chart(fig_scatter, width='stretch')

    # Time series decomposition
    if show_decomposition:
        st.subheader("🔍 Time Series Decomposition")

        try:
            # Perform decomposition
            decomposition = seasonal_decompose(df['value'], model='additive', period=seasonal_period)

            # Create subplots for decomposition
            fig_decomp = make_subplots(rows=4, cols=1,
                                     subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                                     shared_xaxes=True)

            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.observed,
                                          mode='lines', name='Observed'), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.trend,
                                          mode='lines', name='Trend'), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.seasonal,
                                          mode='lines', name='Seasonal'), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.resid,
                                          mode='lines', name='Residual'), row=4, col=1)

            fig_decomp.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_decomp, width='stretch')

        except Exception as e:
            st.error(f"Decomposition failed: {str(e)}")

    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20))

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, Pandas, NumPy, Plotly, and Statsmodels")

if __name__ == "__main__":
    main()