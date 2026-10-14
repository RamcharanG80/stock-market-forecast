
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Stock Market Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market Forecasting Dashboard")
st.caption(
    "Multi-stock analysis | Technical indicators | "
    "Machine learning | Walk-forward evaluation"
)

st.info(
    "Educational project only. Forecasts are experimental estimates, "
    "not guaranteed future prices or financial advice."
)


# =========================================================
# CONSTANTS
# =========================================================

STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

FEATURES = [
    "Return_1D",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI_14",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
]

MODEL_NAMES = [
    "Naive Baseline",
    "Linear Regression",
    "Random Forest",
]


# =========================================================
# DATA DOWNLOAD AND CACHING
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_stock_data(ticker, period):
    """Download and cache daily adjusted stock prices."""

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(
            f"No stock data was returned for {ticker}. "
            "Check the ticker or try again later."
        )

    # Handle yfinance MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]
    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    for column in required:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=["Close"])

    if len(data) < 80:
        raise ValueError(
            f"Only {len(data)} price rows are available for {ticker}. "
            "Choose a longer history period."
        )

    return data


# =========================================================
# TECHNICAL INDICATORS AND FEATURES
# =========================================================

@st.cache_data(show_spinner=False)
def build_features(data):
    """Create indicators and next-session prediction targets."""

    df = data.copy()
    close = df["Close"]

    df["Return_1D"] = close.pct_change()
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI using rolling average gains and losses.
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    relative_strength = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # If there were no losses, RSI is conventionally 100.
    df.loc[(loss == 0) & (gain > 0), "RSI_14"] = 100

    for lag in [1, 2, 3, 5]:
        df[f"Lag_{lag}"] = close.shift(lag)

    # Target is the next trading session's close.
    df["Target"] = close.shift(-1)

    return df.replace([np.inf, -np.inf], np.nan)


# =========================================================
# MODEL METRICS
# =========================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1
            else np.nan
        ),
    }


def make_linear_model():
    return make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


# =========================================================
# WALK-FORWARD MODEL EVALUATION
# =========================================================

@st.cache_data(show_spinner=False)
def evaluate_models(ticker, period, folds):
    """
    Expanding-window walk-forward evaluation.
    Cached by ticker, history period and fold count.
    """

    raw = download_stock_data(ticker, period)
    df = build_features(raw)

    supervised = df.dropna(
        subset=FEATURES + ["Target"]
    ).copy()

    if len(supervised) < 100:
        raise ValueError(
            "Not enough usable rows for model evaluation. "
            "Select a longer history period."
        )

    # Keep an initial training window and evaluate later rows.
    first_test = int(len(supervised) * 0.70)
    test_positions = np.arange(first_test, len(supervised))
    fold_positions = np.array_split(test_positions, folds)

    fold_rows = []
    prediction_rows = []

    for fold_number, positions in enumerate(
        fold_positions, start=1
    ):
        if len(positions) == 0:
            continue

        start = int(positions[0])

        train = supervised.iloc[:start]
        test = supervised.iloc[positions]

        X_train = train[FEATURES]
        y_train = train["Target"]
        X_test = test[FEATURES]
        y_test = test["Target"]

        # Baseline predicts tomorrow's close will equal today's close.
        predictions = {
            "Naive Baseline": test["Close"].to_numpy()
        }

        linear = make_linear_model()
        linear.fit(X_train, y_train)
        predictions["Linear Regression"] = linear.predict(X_test)

        forest = make_random_forest()
        forest.fit(X_train, y_train)
        predictions["Random Forest"] = forest.predict(X_test)

        for model_name, predicted in predictions.items():
            metrics = calculate_metrics(y_test, predicted)

            fold_rows.append({
                "Fold": fold_number,
                "Model": model_name,
                **metrics,
            })

            for i, (actual, estimate) in enumerate(
                zip(y_test.to_numpy(), predicted)
            ):
                prediction_rows.append({
                    "Fold": fold_number,
                    "Model": model_name,
                    "Date": test.index[i],
                    "Actual_Next_Close": actual,
                    "Predicted_Next_Close": estimate,
                    "Absolute_Error": abs(actual - estimate),
                })

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    if folds_df.empty:
        raise ValueError(
            "Walk-forward evaluation could not create any folds."
        )

    # Average metrics across folds.
    summary = (
        folds_df.groupby("Model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2=("R2", "mean"),
        )
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    return summary, folds_df, predictions_df


# =========================================================
# NEXT-SESSION FORECASTS
# =========================================================

@st.cache_data(show_spinner=False)
def forecast_next_session(ticker, period):
    """Fit models on all available labelled rows and forecast next close."""

    raw = download_stock_data(ticker, period)
    df = build_features(raw)

    supervised = df.dropna(
        subset=FEATURES + ["Target"]
    ).copy()

    latest = df.dropna(subset=FEATURES).iloc[[-1]]

    if len(supervised) < 80 or latest.empty:
        raise ValueError(
            "Not enough usable data to generate a forecast."
        )

    X = supervised[FEATURES]
    y = supervised["Target"]
    latest_X = latest[FEATURES]

    estimates = {
        "Naive Baseline": float(latest["Close"].iloc[0])
    }

    linear = make_linear_model()
    linear.fit(X, y)
    estimates["Linear Regression"] = float(
        linear.predict(latest_X)[0]
    )

    forest = make_random_forest()
    forest.fit(X, y)
    estimates["Random Forest"] = float(
        forest.predict(latest_X)[0]
    )

    rows = []
    last_close = float(latest["Close"].iloc[0])

    for model, estimate in estimates.items():
        rows.append({
            "Ticker": ticker,
            "Last_Observed_Date": latest.index[0],
            "Last_Close": last_close,
            "Model": model,
            "Forecast_Next_Close": estimate,
            "Change_Percent": (
                (estimate / last_close - 1) * 100
                if last_close else np.nan
            ),
        })

    return pd.DataFrame(rows)


# =========================================================
# OPTIONAL LSTM
# =========================================================

def forecast_lstm(data, lookback=30, epochs=8):
    """
    Optional LSTM forecast.
    TensorFlow is imported only when this function is called.
    """

    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)

    prices = data["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(prices) < lookback + 60:
        raise ValueError(
            "Not enough price history for the optional LSTM."
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.asarray(X, dtype=np.float32).reshape(
        -1, lookback, 1
    )
    y = np.asarray(y, dtype=np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=2,
        restore_best_weights=True,
    )

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping],
    )

    last_sequence = scaled[-lookback:].reshape(
        1, lookback, 1
    )

    scaled_forecast = model.predict(
        last_sequence, verbose=0
    )

    result = scaler.inverse_transform(
        scaled_forecast
    )[0, 0]

    tf.keras.backend.clear_session()

    return float(result)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Dashboard Settings")

selected_names = st.sidebar.multiselect(
    "Stocks for comparison",
    options=list(STOCKS.keys()),
    default=["Reliance Industries", "TCS", "Infosys"],
)

custom_ticker = st.sidebar.text_input(
    "Or enter a custom Yahoo Finance ticker",
    placeholder="e.g. RELIANCE.NS",
).strip().upper()

period = st.sidebar.selectbox(
    "Historical data period",
    options=["6mo", "1y", "2y", "5y", "10y"],
    index=1,
)

folds = st.sidebar.slider(
    "Walk-forward folds",
    min_value=2,
    max_value=5,
    value=3,
)

use_lstm = st.sidebar.checkbox(
    "Include optional LSTM forecast",
    value=False,
)

tickers = [STOCKS[name] for name in selected_names]

if custom_ticker:
    tickers.append(custom_ticker)

tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.warning("Select at least one stock in the sidebar.")
    st.stop()

primary_ticker = (
    custom_ticker if custom_ticker else tickers[0]
)

st.sidebar.caption(
    "Cached downloads expire after 15 minutes. "
    "Changing the ticker, period or fold count triggers new work."
)


# =========================================================
# LOAD PRIMARY DATA
# =========================================================

try:
    with st.spinner(f"Loading {primary_ticker}..."):
        primary_data = download_stock_data(
            primary_ticker, period
        )
        primary_features = build_features(primary_data)

except Exception as exc:
    st.error(f"Could not load stock data: {exc}")
    st.stop()


# =========================================================
# OVERVIEW METRICS
# =========================================================

last_close = float(primary_data["Close"].iloc[-1])
previous_close = (
    float(primary_data["Close"].iloc[-2])
    if len(primary_data) > 1 else last_close
)

change_pct = (
    (last_close / previous_close - 1) * 100
    if previous_close else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Selected stock",
    primary_ticker,
)

col2.metric(
    "Latest close",
    f"₹{last_close:,.2f}",
)

col3.metric(
    "Daily change",
    f"{change_pct:+.2f}%",
)

col4.metric(
    "Historical sessions",
    f"{len(primary_data):,}",
)


# =========================================================
# TABS
# =========================================================

overview_tab, indicators_tab, evaluation_tab, forecast_tab, downloads_tab = (
    st.tabs([
        "📊 Overview",
        "📉 Technical Indicators",
        "🧪 Model Evaluation",
        "🔮 Forecasts",
        "⬇️ Downloads",
    ])
)


# =========================================================
# OVERVIEW TAB
# =========================================================

with overview_tab:
    st.subheader("Historical closing prices")

    st.line_chart(
        primary_data["Close"].rename("Close"),
        height=350,
    )

    st.subheader("Compare multiple stocks")

    if len(tickers) > 1:
        normalized = {}

        with st.spinner("Loading comparison data..."):
            for ticker in tickers:
                try:
                    stock_data = download_stock_data(
                        ticker, period
                    )

                    close = stock_data["Close"].dropna()

                    # Normalize to 100 at the first observation.
                    normalized[ticker] = (
                        close / close.iloc[0] * 100
                    )

                except Exception as exc:
                    st.warning(
                        f"Skipped {ticker}: {exc}"
                    )

        if normalized:
            comparison = pd.DataFrame(normalized)
            st.line_chart(comparison, height=350)
            st.caption(
                "Indexed to 100 at each stock's first available "
                "observation; this compares relative price movement, "
                "not absolute share prices."
            )
    else:
        st.caption(
            "Select two or more stocks in the sidebar to compare them."
        )

    st.subheader("Recent market data")
    st.dataframe(
        primary_data.tail(15).sort_index(ascending=False),
        use_container_width=True,
    )


# =========================================================
# TECHNICAL INDICATORS TAB
# =========================================================

with indicators_tab:
    st.subheader("Moving averages")

    moving_averages = primary_features[
        ["Close", "SMA_10", "SMA_20", "SMA_50", "EMA_20"]
    ].dropna()

    st.line_chart(moving_averages, height=350)

    st.subheader("Relative Strength Index (RSI)")

    st.line_chart(
        primary_features["RSI_14"].dropna(),
        height=220,
    )

    st.caption(
        "RSI is a momentum indicator. Values near 70 or 30 are "
        "commonly watched, but they do not guarantee reversals."
    )

    st.subheader("Daily returns")

    st.line_chart(
        primary_features["Return_1D"].dropna(),
        height=220,
    )

    st.subheader("Indicator values")

    indicator_columns = [
        "Close",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "Return_1D",
    ]

    st.dataframe(
        primary_features[indicator_columns]
        .tail(30)
        .sort_index(ascending=False),
        use_container_width=True,
    )


# =========================================================
# MODEL EVALUATION TAB
# =========================================================

with evaluation_tab:
    st.subheader("Walk-forward evaluation")

    st.write(
        "Models are trained on earlier observations and tested on "
        "later observations. Metrics are averaged across folds."
    )

    run_evaluation = st.button(
        "Run / refresh model evaluation",
        type="primary",
    )

    if run_evaluation:
        # Clear only the evaluation cache when explicitly requested.
        evaluate_models.clear()

    try:
        with st.spinner(
            "Evaluating models. The first run may take a little longer..."
        ):
            summary, fold_results, prediction_results = (
                evaluate_models(
                    primary_ticker, period, folds
                )
            )

        st.success("Model evaluation is ready.")

        best_model = summary.iloc[0]["Model"]
        best_mae = summary.iloc[0]["MAE"]

        col1, col2 = st.columns(2)
        col1.metric("Lowest average MAE", best_model)
        col2.metric("Average MAE", f"{best_mae:.2f}")

        st.subheader("Model comparison")
        st.dataframe(
            summary.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            summary.set_index("Model")[["MAE", "RMSE"]],
            height=300,
        )

        st.subheader("Metrics by fold")
        st.dataframe(
            fold_results.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Actual versus predicted")
        model_to_plot = st.selectbox(
            "Select model",
            options=summary["Model"].tolist(),
            key="evaluation_model",
        )

        plot_df = prediction_results[
            prediction_results["Model"] == model_to_plot
        ].copy()

        if not plot_df.empty:
            plot_df["Date"] = pd.to_datetime(plot_df["Date"])
            plot_df = plot_df.set_index("Date")

            st.line_chart(
                plot_df[
                    ["Actual_Next_Close", "Predicted_Next_Close"]
                ],
                height=350,
            )

        st.session_state["evaluation_summary"] = summary
        st.session_state["evaluation_folds"] = fold_results
        st.session_state["evaluation_predictions"] = prediction_results

    except Exception as exc:
        st.error(f"Model evaluation failed: {exc}")


# =========================================================
# FORECAST TAB
# =========================================================

with forecast_tab:
    st.subheader("Next-session price estimates")

    st.caption(
        "These are model estimates for the next trading session, "
        "not guaranteed prices."
    )

    if st.button(
        "Generate next-session forecasts",
        type="primary",
    ):
        try:
            with st.spinner("Training forecasting models..."):
                forecast_table = forecast_next_session(
                    primary_ticker, period
                )

            if use_lstm:
                with st.spinner(
                    "Training optional LSTM. This may take longer..."
                ):
                    lstm_estimate = forecast_lstm(primary_data)

                last = float(primary_data["Close"].iloc[-1])

                lstm_row = pd.DataFrame([{
                    "Ticker": primary_ticker,
                    "Last_Observed_Date": primary_data.index[-1],
                    "Last_Close": last,
                    "Model": "LSTM",
                    "Forecast_Next_Close": lstm_estimate,
                    "Change_Percent": (
                        (lstm_estimate / last - 1) * 100
                        if last else np.nan
                    ),
                }])

                forecast_table = pd.concat(
                    [forecast_table, lstm_row],
                    ignore_index=True,
                )

            st.session_state["forecast_table"] = forecast_table

        except ImportError:
            st.error(
                "TensorFlow is required for LSTM. Install it with "
                "`python -m pip install tensorflow`, then restart the app."
            )
        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")

    if "forecast_table" in st.session_state:
        forecast_table = st.session_state["forecast_table"]

        st.dataframe(
            forecast_table.round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            forecast_table.set_index("Model")[
                ["Forecast_Next_Close"]
            ],
            height=300,
        )

        csv_bytes = forecast_table.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download forecasts CSV",
            data=csv_bytes,
            file_name="next_session_forecasts.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Click 'Generate next-session forecasts' to train the "
            "models and display their estimates."
        )


# =========================================================
# DOWNLOADS TAB
# =========================================================

with downloads_tab:
    st.subheader("Download reports")

    st.write(
        "Download technical indicators, historical prices, model "
        "metrics, predictions and forecasts."
    )

    st.download_button(
        "Download historical prices CSV",
        data=primary_data.to_csv().encode("utf-8"),
        file_name=f"{primary_ticker}_historical_prices.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download technical indicators CSV",
        data=primary_features.to_csv().encode("utf-8"),
        file_name=f"{primary_ticker}_technical_indicators.csv",
        mime="text/csv",
    )

    if "evaluation_summary" in st.session_state:
        st.download_button(
            "Download model comparison CSV",
            data=st.session_state["evaluation_summary"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download fold metrics CSV",
            data=st.session_state["evaluation_folds"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="walk_forward_fold_metrics.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download test predictions CSV",
            data=st.session_state["evaluation_predictions"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="walk_forward_predictions.csv",
            mime="text/csv",
        )

    if "forecast_table" in st.session_state:
        st.download_button(
            "Download forecast CSV",
            data=st.session_state["forecast_table"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="next_session_forecasts.csv",
            mime="text/csv",
        )

    if st.button("Create ZIP report"):
        try:
            report_files = {
                "historical_prices.csv": primary_data.to_csv(),
                "technical_indicators.csv": primary_features.to_csv(),
            }

            if "evaluation_summary" in st.session_state:
                report_files["model_comparison.csv"] = (
                    st.session_state["evaluation_summary"].to_csv(
                        index=False
                    )
                )
                report_files["fold_metrics.csv"] = (
                    st.session_state["evaluation_folds"].to_csv(
                        index=False
                    )
                )
                report_files["predictions.csv"] = (
                    st.session_state["evaluation_predictions"].to_csv(
                        index=False
                    )
                )

            if "forecast_table" in st.session_state:
                report_files["forecasts.csv"] = (
                    st.session_state["forecast_table"].to_csv(
                        index=False
                    )
                )

            buffer = BytesIO()

            with ZipFile(
                buffer, "w", compression=ZIP_DEFLATED
            ) as zip_file:
                for filename, content in report_files.items():
                    zip_file.writestr(filename, content)

            st.download_button(
                "Download ZIP report",
                data=buffer.getvalue(),
                file_name=f"{primary_ticker}_stock_reports.zip",
                mime="application/zip",
            )

        except Exception as exc:
            st.error(f"Could not create ZIP report: {exc}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Model results depend on the selected stock, date range, "
    "data availability and evaluation settings. Past performance "
    "does not guarantee future results."
)
