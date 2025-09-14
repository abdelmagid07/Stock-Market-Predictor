import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("📈 Stock Analysis & Prediction App")
st.write("Choose between short-term (5-day) predictions or long-term trend analysis.")

# Stock selection
stocks = ("AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA", "INTC", "AMD", "IBM")
selected_stock = st.selectbox("Select a Stock", stocks)

# Create tabs for short-term and long-term analysis
tab1, tab2 = st.tabs(["📊 5-Day Prediction", "📈 Long-Term Trends"])

# Short-term prediction
with tab1:
    st.header("5-Day Stock Movement Prediction")
    st.write("Uses Random Forest ML to predict if stock will go UP or DOWN in 5 trading days.")
    
    try:
        # Load stock data
        with st.spinner(f"Loading data for {selected_stock}..."):
            data = yf.Ticker(selected_stock)
            df = data.history(period="max")
        
        if df.empty:
            st.error(f"No data available for {selected_stock}")
        else:
            # Remove unnecessary columns
            df.drop(columns=[col for col in ["Dividends", "Stock Splits"] if col in df.columns], inplace=True)

            # Target variable: 1 if price goes up in 5 days, else 0
            df["Price_in_5_days"] = df["Close"].shift(-5)
            df["Target"] = (df["Price_in_5_days"] > df["Close"]).astype(int)

            # Use data from 1990 onwards
            df = df.loc["1990-01-01":].copy()

            # Features
            predictors = ["Close", "Volume", "Open", "High", "Low"]
            horizons = [2, 5, 60, 250, 1000]

            with st.spinner("Creating features..."):
                for h in horizons:
                    if len(df) > h:
                        rolling_avg = df.rolling(h).mean()
                        df[f"Close_Ratio_{h}"] = df["Close"] / rolling_avg["Close"]
                        df[f"Trend_{h}"] = df["Target"].shift(1).rolling(h).sum()
                        predictors += [f"Close_Ratio_{h}", f"Trend_{h}"]

            # Clean data
            df_clean = df.dropna()

            if len(df_clean) < 200:
                st.error(f"Not enough data for {selected_stock}. Need at least 200 rows, have {len(df_clean)}")
            else:
                # Train/test split: last 100 days for testing
                train = df_clean.iloc[:-100]
                test = df_clean.iloc[-100:]

                # Train Random Forest
                with st.spinner("Training model..."):
                    model = RandomForestClassifier(
                        n_estimators=500,
                        max_depth=15,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(train[predictors], train["Target"])

                # Prediction for most recent data
                latest = df_clean.iloc[-1:][predictors]
                pred = model.predict(latest)[0]
                confidence = max(model.predict_proba(latest)[0]) * 100
                direction = "UP" if pred == 1 else "DOWN"

                # Show results
                st.subheader(f"Prediction for {selected_stock}")
                st.success(f"The model predicts **{selected_stock}** will go **{direction}** in 5 trading days.")
                st.info(f"Confidence: **{confidence:.1f}%**")

                # Model performance on test set
                st.subheader("Model Performance on Recent Data")
                test_predictions = model.predict(test[predictors])
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Accuracy", f"{accuracy_score(test['Target'], test_predictions):.1%}")
                with col2:
                    st.metric("Precision", f"{precision_score(test['Target'], test_predictions, zero_division=0):.1%}")
                with col3:
                    baseline = max(test["Target"].mean(), 1 - test["Target"].mean())
                    st.metric("Baseline", f"{baseline:.1%}")

                # Recent predictions vs actual results
                with st.expander("Recent predictions vs actual results"):
                    recent_results = pd.DataFrame({
                        'Date': test.index[-20:],
                        'Actual': test["Target"].iloc[-20:].values,
                        'Predicted': test_predictions[-20:],
                        'Close Price': test["Close"].iloc[-20:].values
                    })
                    recent_results['Actual_Direction'] = recent_results['Actual'].map({1: 'UP', 0: 'DOWN'})
                    recent_results['Predicted_Direction'] = recent_results['Predicted'].map({1: 'UP', 0: 'DOWN'})
                    recent_results['Correct?'] = recent_results['Actual'] == recent_results['Predicted']
                    st.dataframe(recent_results[['Date', 'Close Price', 'Actual_Direction', 'Predicted_Direction', 'Correct?']])

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.info("Try selecting a different stock or reloading the page.")

# Long-term trend analysis
with tab2:
    st.header("Long-Term Trend Analysis & Prediction")
    st.write("Uses Facebook Prophet to forecast long-term price trends.")

    n_years = st.slider('Years of prediction:', 1, 5)
    period = n_years * 365

    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    try:
        with st.spinner('Loading long-term data...'):
            data_long = load_data(selected_stock)

        if data_long.empty:
            st.error(f"No long-term data available for {selected_stock}")
        else:
            st.subheader('Recent Raw Data')
            st.write(data_long.tail())

            # Prepare data for Prophet
            if isinstance(data_long.columns, pd.MultiIndex):
                data_long.columns = [col[0] if isinstance(col, tuple) else col for col in data_long.columns]

            y_col = 'Adj Close' if 'Adj Close' in data_long.columns else 'Close' if 'Close' in data_long.columns else None
            if y_col:
                df_train = data_long[['Date', y_col]].rename(columns={'Date': 'ds', y_col: 'y'})
                df_train['ds'] = pd.to_datetime(df_train['ds'])
                df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
                df_train = df_train.dropna()

                with st.spinner('Training Prophet model...'):
                    m = Prophet()
                    m.fit(df_train)
                    future = m.make_future_dataframe(periods=period)
                    forecast = m.predict(future)

                st.subheader('Forecast Data')
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                st.subheader(f'Price Forecast for {n_years} Year(s)')
                fig1 = plot_plotly(m, forecast)
                fig1.update_layout(title=f"{selected_stock} Price Prediction")
                st.plotly_chart(fig1, use_container_width=True)

                st.subheader("Forecast Components")
                st.write("Shows trend, weekly patterns, and yearly seasonal patterns.")
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error in long-term analysis: {str(e)}")
        st.info("Try selecting a different stock or adjusting the prediction period.")
