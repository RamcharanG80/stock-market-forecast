# 📈 Stock Market Forecasting Dashboard

A Python-based stock market analysis and forecasting dashboard built using **Streamlit**. The application helps users explore historical stock prices, analyze technical indicators, evaluate forecasting models, and generate predictions for upcoming closing prices.

## 🚀 Features

- **Stock Market Analysis:** View historical stock price data for selected stocks.
- **Technical Indicators:** Analyze SMA, EMA, RSI, and daily returns.
- **Stock Price Forecasting:** Generate forecasts using Linear Regression and Random Forest models.
- **Naive Baseline:** Compare machine learning models against a simple baseline forecasting method.
- **Model Evaluation:** Evaluate forecasting performance using historical walk-forward testing.
- **LSTM Forecasting:** Optionally use an LSTM deep learning model when TensorFlow is installed.
- **Today's Close Prediction:** Estimate the closing price for the current trading session based on available completed daily data.
- **Actual Closing Price Check:** Check whether the data provider has published today's actual closing price.
- **Data Visualization:** Explore stock prices, indicators, and model performance through interactive charts.
- **Download Reports:** Export forecast results and supporting data as CSV files or a ZIP report.

## 🛠️ Technologies Used

- Python
- Streamlit
- yfinance
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- TensorFlow (optional, for LSTM forecasting)

## 📂 Project Structure

```text
stock-market-forecast/
│
├── app.py              # Main Streamlit application
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies (if included)
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/RamcharanG80/stock-market-forecast.git
```

### 2. Navigate to the project directory

```bash
cd stock-market-forecast
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install the required libraries

```bash
pip install streamlit yfinance pandas numpy scikit-learn matplotlib
```

For optional LSTM forecasting, install TensorFlow separately:

```bash
pip install tensorflow
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## 📊 Forecasting Models

The dashboard supports the following forecasting approaches:

| Model | Purpose |
|---|---|
| Naive Baseline | Provides a simple reference forecast |
| Linear Regression | Models relationships between historical features and future prices |
| Random Forest | Captures nonlinear relationships in historical market data |
| LSTM (Optional) | Uses a recurrent neural network for sequential price forecasting |

The models can be evaluated using walk-forward testing, where predictions are generated using historical data available at each evaluation step.

## 🎯 Today's Closing Price Prediction

The dashboard includes a feature to estimate the current trading session's closing price using historical daily data and machine learning models.

It also provides an option to check the actual closing price after the market closes and once the data provider makes the latest data available.

**Note:** The predicted price is an estimate, not a guaranteed market outcome. Actual closing-price availability depends on the data provider and may be delayed. The application is intended for educational and research purposes, not financial advice.

## 📉 Data Source

Historical stock market data is retrieved through the `yfinance` Python library. Data availability, accuracy, and update timing depend on the provider.

## 🔮 Future Improvements

- Add more forecasting models and performance metrics.
- Improve prediction accuracy through feature engineering and model tuning.
- Add portfolio-level analysis and stock comparisons.
- Include more interactive charts and downloadable evaluation reports.
- Improve deployment and error handling.

## 👨‍💻 Author

**Ram Charan G.**

GitHub: [RamcharanG80](https://github.com/RamcharanG80)

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. Stock market predictions are uncertain, and this application does not provide financial or investment advice. Always conduct independent research before making investment decisions.
