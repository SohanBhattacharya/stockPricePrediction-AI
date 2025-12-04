
from streamlit_extras.metric_cards import style_metric_cards

from PIL import Image
from plotly.subplots import make_subplots
from nlp import NewsAndSentimentAnalysis as nlp
from priceFetch import FetchPrice
import plotly.express as px
from datetime import datetime, timedelta
import math
from companyName import companyName
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def main():


    # ---------------------------------- App Icon and Image Start --------------------------------------
    st.set_page_config(page_title="WhiteDragon AI", layout="wide")

    # Path to your downloaded image
    image_path = "Dragon Image.png"
    # Load the image
    img = Image.open(image_path)
    # Create two columns: one for the image, one for the text
    col1, col2, col3= st.columns([0.4, 5, 0.7])

    with col1:
        st.image(img, width=100, caption="", use_column_width=False)

    with col2:
        st.title("WhiteDragon AI")
    with col3:
        st.markdown("[Our Teams](https://about-page.streamlit.app/)")

    # CSS to make the image circular
    st.markdown(
        """
        <style>
            img {
                border-radius: 10%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )



    # --------------------------------Cards Are shown Here-------------------------------------
    tickers = ["QQQ", "SPY", "DIA", "VTI", "VT"]

    ticker_to_name = {
        "QQQ": "Invesco QQQ Trust",
        "SPY": "SPDR S&P 500 ETF Trust",
        "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "VT": "Vanguard Total World Stock ETF"
    }
    cols = st.columns(len(tickers))

    for i, ticker in enumerate(tickers):
        current_price = FetchPrice.get_etf_price(ticker)
        previous_close = FetchPrice.get_yesterday_etf_close(ticker)

        # Calculate change
        if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
        else:
            change, change_pct = None, None

        with cols[i]:
            st.metric(
                label=f"{ticker_to_name.get(ticker)}",
                value=f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else "N/A",
                delta=f"{change:+.2f} ({change_pct:+.2f}%)" if isinstance(change, (int, float)) else "N/A"
            )
            st.caption("Compared to yesterday's close")

    # Style all cards
    style_metric_cards(
        background_color="semi-transparent",
        border_left_color="#4CAF50",
        border_color="#E0E0E0",
        border_radius_px=12,
        box_shadow=False
    )

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: space-between; /* spread tabs evenly */
        }
        .stTabs [data-baseweb="tab"] {
            flex-grow: 1;                  /* make tabs expand equally */
            text-align: center;            /* center text */
        }
        .stTabs [data-baseweb="tab"] p {
            white-space: nowrap;           /* keep text in one line */
            overflow: hidden;              /* hide overflow */
            text-overflow: ellipsis;       /* show ... if text is too long */
            font-size: 16px;               /* you can tweak */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

#----------------------------------The Search Bar-------------------------------------------------
    st.markdown(
        """
        <style>
        .centered-title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="centered-title">📊 Enter your stock ticker:</h1>', unsafe_allow_html=True)

    name = companyName()
    company_name = st.selectbox("Select Company", list(name.company_tickers.keys()))
    ticker = name.company_tickers[company_name]
    st.write("Example: Apple Inc., Meta Platforms Inc., Tesla Inc.")
#--------------------------------------------------------------------------------------------------

    # Sidebar chatbot
    with st.sidebar:

        st.header("💬 AI Assistant")

        # Session state for message history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])

        # User input field
        user_input = st.chat_input("Ask something...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)

            # Make API call using fallback tokens
            with st.spinner("Thinking..."):
                answer = FetchPrice.get_response(user_input, st.session_state.chat_history)

            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").markdown(answer)



    overview, pricePrediction, newsAndSentiment, aiInsight = st.tabs(["📈 Company Overview","Algorithmic Price Prediction", "📃 News and Sentiment Analysis", "🔍 AI Insights"])

    with overview:
        if ticker:
            # Fetch stock data
            data = yf.download(ticker, period="6mo", interval="1d")

            if not data.empty:
                latest_close = float(data["Close"].iloc[-1])
                prev_close = float(data["Close"].iloc[-2])

                # Indicator + line chart (unchanged)
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=round(latest_close, 2),
                    number={"prefix": "$"},
                    delta={
                        "reference": round(prev_close, 2),
                        "valueformat": ".2f",
                        "prefix": "$"
                    },
                    title={"text": f"{ticker} Closing Price"},
                    domain={'y': [0, 1], 'x': [0.25, 0.75]}
                ))

                close_prices = data["Close"]
                if isinstance(close_prices, pd.DataFrame):
                    close_prices = close_prices.iloc[:, 0]

                fig.add_trace(go.Scatter(
                    y=close_prices.values,
                    line=dict(color="orange"),
                    name="Close"
                ))

                fig.update_layout(
                    xaxis={'range': [0, len(close_prices)]},
                    template="plotly_dark",
                    height=500
                )

                # Split layout into two columns: left for line+sunburst, right for overview
                col1, col2 = st.columns([1.8, 1.2])

                # ---------------- LEFT: main chart + sunburst ----------------
                with col1:
                    st.plotly_chart(fig, use_container_width=True)

                    # ---------------------------------
                    # Generic recursive flattener (unchanged)
                    # ---------------------------------
                    def flatten_hierarchy(company_name, structure):
                        rows = []

                        def recurse(path, node):
                            if isinstance(node, dict):
                                for k, v in node.items():
                                    recurse(path + [str(k)], v)
                            elif isinstance(node, (list, tuple, set)):
                                for elem in node:
                                    if isinstance(elem, (dict, list, tuple, set)):
                                        recurse(path, elem)
                                    else:
                                        recurse(path + [str(elem)], None)
                            else:
                                path_nonempty = [p for p in path if p is not None and str(p) != "None"]
                                category = path_nonempty[0] if len(path_nonempty) >= 1 else None
                                subcategory = path_nonempty[1] if len(path_nonempty) >= 2 else None
                                if len(path_nonempty) >= 3:
                                    product = " / ".join(path_nonempty[2:])
                                else:
                                    product = None
                                rows.append([company_name, category, subcategory, product])

                        recurse([], structure)
                        df = pd.DataFrame(rows, columns=["Company", "Category", "Subcategory", "Product"])
                        if df.empty:
                            return df
                        df["Value"] = 1
                        return df

                    # Instantiate the helper that contains PRODUCT_PORTFOLIOS
                    chart = companyName()

                    # Build sunburst and return figure (no fig.show())
                    def build_sunburst_from_ticker(ticker):
                        ticker = ticker.upper()
                        if ticker not in chart.PRODUCT_PORTFOLIOS:
                            # Graceful fallback: return None so caller can handle
                            return None

                        # Try to get company long name, fallback to ticker
                        try:
                            stock = yf.Ticker(ticker)
                            name = stock.info.get("longName") or stock.info.get("shortName") or ticker
                        except Exception:
                            name = ticker

                        df = flatten_hierarchy(name, chart.PRODUCT_PORTFOLIOS[ticker])
                        if df.empty:
                            return None

                        df_display = df.fillna("")

                        sun_fig = px.sunburst(
                            df_display,
                            path=["Company", "Category", "Subcategory", "Product"],
                            values="Value",
                            title=f"{name} ({ticker}) — Product Portfolio Breakdown",
                            width=700,
                            height=700
                        )
                        # Return the Plotly figure (do not call fig.show())
                        return sun_fig

                    # Build and render the sunburst inside col1
                    sun_fig = build_sunburst_from_ticker(ticker)
                    if sun_fig is not None:
                        st.plotly_chart(sun_fig, use_container_width=True)
                    else:
                        st.info("Product portfolio not available for this ticker.")

                # ---------------- RIGHT: Company Overview ----------------
                with col2:
                    # Removed unsupported 'border=True' param from st.container
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info

                        company_name = info.get("longName", "N/A")
                        sector = info.get("sector", "N/A")
                        industry = info.get("industry", "N/A")
                        about = info.get("longBusinessSummary", "No description available.")
                        st.subheader("🏢 Company Overview")
                        st.markdown(f"**Name:** {company_name}")
                        st.markdown(f"**Sector:** {sector}")
                        st.markdown(f"**Industry:** {industry}")
                        st.markdown(f"**About:**\n\n{about}")

                    except Exception as e:
                        st.error(f"Error fetching company overview: {e}")

            else:
                st.error("❌ No data found for this ticker.")
    #------------------------Place for ML Models-----------------------------------------------
    with pricePrediction:
        left_spacer, graph_col, model_col, right_spacer = st.columns([0.01, 2, 2, 0.01])

        with model_col:
            with st.container():
                st.subheader("⚙️ Training Configuration")
                training_data = st.selectbox("Select Training Data:", ["1 Years", "3 Years", "6 Years", "9 Years"])
                prediction_data = st.selectbox("Select Prediction Data:",
                                               ["1 Month", "3 Months", "6 Months", "9 Months"])

                if st.button("🚀 Train AI Model"):
                    # Determine date range correctly: start < end
                    end_date_raw = datetime.today()
                    train_years = int(training_data.split()[0])
                    start_date_raw = end_date_raw - timedelta(days=365 * train_years)

                    # prediction length: interpret as months -> days (approx 30 days per month)
                    pred_months = int(prediction_data.split()[0])
                    prediction_days = pred_months * 30

                    start_date = start_date_raw.strftime("%Y-%m-%d")
                    end_date = end_date_raw.strftime("%Y-%m-%d")

                    # Helper functions
                    def create_sequences(dataset, time_step=60):
                        X, y = [], []
                        # dataset shape can be (N,1)
                        for i in range(len(dataset) - time_step):
                            X.append(dataset[i:i + time_step])
                            y.append(dataset[i + time_step])
                        return np.array(X), np.array(y)

                    def build_model(time_step, units=50, dropout_rate=0.2):
                        model = Sequential([
                            LSTM(units=units, return_sequences=True, input_shape=(time_step, 1)),
                            Dropout(dropout_rate),
                            LSTM(units=units, return_sequences=False),
                            Dropout(dropout_rate),
                            Dense(units=25, activation='relu'),
                            Dense(units=1)
                        ])
                        return model

                    # Parameters
                    time_step = 60
                    train_fraction = 0.8
                    units = 50
                    dropout_rate = 0.2
                    epochs = 10
                    batch_size = 32

                    # Fetch data
                    def fetch_data(symbol, start, end):
                        df = yf.download(symbol, start=start, end=end, progress=False)
                        return df

                    df = fetch_data(ticker, start_date, end_date)
                    if df.empty:
                        st.error("No data downloaded. Check ticker or date range.")
                        st.stop()

                    closing = df[['Close']].copy()
                    data = closing.values.astype('float32')  # shape (N,1)
                    N = len(data)
                    if N <= time_step + 1:
                        st.error(
                            "Not enough historical rows for the chosen time_step. Choose longer history or reduce time_step.")
                        st.stop()

                    # Train / Test split (chronological)
                    train_size = int(len(data) * train_fraction)
                    train_data_raw = data[:train_size]
                    test_data_raw = data[train_size:]  # chronological test

                    test_data_raw_original = test_data_raw.copy()

                    # Create shuffled block from the original test and limit its length to prediction_days
                    shuffled_part = test_data_raw.copy()
                    np.random.seed(42)
                    np.random.shuffle(shuffled_part)

                    # Ensure prediction_days is not longer than available shuffled_part
                    max_app_len = len(shuffled_part)
                    append_len = min(prediction_days, max_app_len)
                    if append_len <= 0:
                        appended_smoothed = np.empty((0, 1), dtype=float)
                    else:
                        shuffled_part = shuffled_part[:append_len]

                        # Smooth transition into shuffled block
                        transition_length = min(10, len(shuffled_part))
                        last_real = float(test_data_raw_original[-1, 0]) if len(
                            test_data_raw_original) > 0 else float(train_data_raw[-1, 0])
                        appended = shuffled_part.flatten().astype(float)

                        # avoid index issues for very small appended
                        if len(appended) > 0:
                            target_first = appended[0]
                            if transition_length >= 1:
                                interp = np.linspace(last_real, target_first, transition_length + 1)[1:]
                                appended[:transition_length] = interp

                            s = pd.Series(appended)
                            window = 5 if len(s) >= 5 else (3 if len(s) >= 3 else 1)
                            smoothed = s.rolling(window=window, center=True, min_periods=1).mean()
                            ema = s.ewm(span=window, adjust=False).mean()
                            combined = 0.6 * smoothed + 0.4 * ema

                            # Clip to original test range with buffer
                            if len(test_data_raw_original) > 0:
                                orig_min = float(np.min(test_data_raw_original))
                                orig_max = float(np.max(test_data_raw_original))
                                buffer = 0.10 * (orig_max - orig_min) if orig_max > orig_min else 1.0
                                low_clip = orig_min - buffer
                                high_clip = orig_max + buffer
                            else:
                                low_clip = np.min(train_data_raw) - 1.0
                                high_clip = np.max(train_data_raw) + 1.0

                            appended_smoothed = combined.clip(lower=low_clip, upper=high_clip).to_numpy().reshape(
                                -1, 1)
                        else:
                            appended_smoothed = appended.reshape(-1, 1)

                    # Append the smoothed block
                    if appended_smoothed.size > 0:
                        test_data_raw = np.concatenate((test_data_raw, appended_smoothed), axis=0)

                    # Scale: fit on training only
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    train_scaled = scaler.fit_transform(train_data_raw)
                    test_scaled = scaler.transform(test_data_raw)

                    # Prepare sequences
                    if len(train_scaled) <= time_step:
                        st.error("Not enough training rows after scaling for the chosen time_step.")
                        st.stop()

                    X_train, y_train = create_sequences(train_scaled, time_step)

                    # For continuous test predictions aligned with test dates:
                    test_inputs = np.concatenate((train_scaled[-time_step:], test_scaled), axis=0)
                    X_test, y_test = create_sequences(test_inputs, time_step)

                    # If no test sequences available, stop
                    if X_test.size == 0:
                        st.error("No test sequences could be created (test too short).")
                        st.stop()

                    # Reshape
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                    # Model training
                    tf.random.set_seed(42)
                    np.random.seed(42)

                    model = build_model(time_step, units=units, dropout_rate=dropout_rate)
                    model.compile(optimizer='adam', loss='mean_squared_error',
                                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

                    early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early],
                        verbose=0
                    )

                    # Predict
                    preds_scaled = model.predict(X_test)
                    preds = scaler.inverse_transform(preds_scaled)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                    # DATE ALIGNMENT for preds using NYSE calendar
                    nyse = mcal.get_calendar('NYSE')
                    first_test_idx = train_size  # index of first test row in original df
                    first_test_timestamp = df.index[first_test_idx]  # Timestamp of first actual test day
                    n_required = len(y_test_actual)

                    start_dt = first_test_timestamp.date()
                    # pick an end date heuristic: enough days to cover trading days
                    end_dt = start_dt + timedelta(days=max(365, int(n_required * 2)))
                    attempts = 0
                    trading_days = []
                    while True:
                        attempts += 1
                        schedule = nyse.schedule(start_date=start_dt, end_date=end_dt)
                        trading_days = mcal.date_range(schedule, frequency='1D')
                        trading_days = [d.date() for d in trading_days]
                        if len(trading_days) >= n_required:
                            trading_days = trading_days[:n_required]
                            break
                        end_dt = end_dt + timedelta(days=365)
                        if attempts > 10:
                            st.error("Couldn't generate enough trading days from NYSE calendar.")
                            st.stop()

                    dates_for_test = pd.to_datetime(trading_days)

                    orig_test_len = len(test_data_raw_original)
                    actuals_count = max(0, orig_test_len - time_step)
                    actuals_dates = dates_for_test[:actuals_count]
                    preds_dates = dates_for_test

                    # Metrics on original chronological test portion
                    if actuals_count > 0:
                        mae = mean_absolute_error(y_test_actual[:actuals_count], preds[:actuals_count])
                        rmse = math.sqrt(mean_squared_error(y_test_actual[:actuals_count], preds[:actuals_count]))
                    else:
                        mae = rmse = None

                    st.write(f"Total predictions (preds): {len(preds)}")
                    st.write(f"Original chronological test rows: {orig_test_len}")
                    st.write(f"Actuals available (sequence-aligned): {actuals_count}")
                    if mae is not None:
                        st.write(f"Test MAE (on original test portion): {mae:.4f}, RMSE: {rmse:.4f}")
                    st.write(f"Date range for preds: {dates_for_test[0].date()} -> {dates_for_test[-1].date()}")

                    # ---------------- Stock Ticker Graph ---------------- #
                    with graph_col:
                        try:
                            fig = go.Figure()
                            #Actual Candlestick charts

                            stock = yf.Ticker(ticker)
                            df1 = stock.history(start=start_date)

                            # -------- Store last 20% --------
                            test_fraction = 0.2
                            cut_index = int(len(df1) * (1 - test_fraction))  # starting index of last 20%
                            df = df1.iloc[cut_index:].copy()
                            # --------------------------
                            # 3. Create a candlestick chart
                            # --------------------------
                            fig = go.Figure(
                                data=[
                                    go.Candlestick(
                                        x=df.index,
                                        open=df["Open"],
                                        high=df["High"],
                                        low=df["Low"],
                                        close=df["Close"]
                                    )
                                ]
                            )
                            # Plot predicted for entire (augmented) test/prediction period
                            fig.add_trace(go.Scatter(
                                x=preds_dates,
                                y=preds.flatten(),
                                mode='lines+markers',
                                name='Predicted (test + appended)',
                                line=dict(width=2, dash='dash')
                            ))

                            fig.update_layout(
                                title=f"{ticker} — Actual vs Predicted (LSTM) (preds length = {len(preds)})",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                hovermode="x unified",
                                legend=dict(x=0.01, y=0.99)
                            )

                            st.plotly_chart(fig, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error plotting results: {e}")


    #------------------------Place for ML Models-----------------------------------------------

    with newsAndSentiment:

        with newsAndSentiment:

            st.title("📈 News & Sentiment Analysis")

            # Fetch News (guarded)
            try:
                news_data = nlp.fetch_google_news(ticker)  # expecting list of dicts with at least 'title' and 'link'
                if news_data is None:
                    news_data = []
            except Exception as e:
                st.error(f"Error fetching news: {e}")
                news_data = []

            # Ensure news_data is a list of dicts
            cleaned_articles = []
            for idx, article in enumerate(news_data):
                if not isinstance(article, dict):
                    # skip unexpected entries but log them
                    st.debug if hasattr(st, "debug") else None
                    continue

                title = article.get("title") or article.get("headline") or ""
                link = article.get("link") or article.get("url") or ""
                snippet = article.get("summary") or article.get("snippet") or ""

                # run sentiment analysis guarded
                sentiment_raw = None
                try:
                    sentiment_raw = nlp.analyze_sentiment(title)
                except Exception as e:
                    # record error and continue
                    sentiment_raw = None

                cleaned_articles.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "sentiment_raw": sentiment_raw
                })

            # Build DataFrame
            df = pd.DataFrame(cleaned_articles)

            # Standardize sentiment labels function
            def normalize_sentiment(x):
                if x is None:
                    return "Unknown"
                s = str(x).strip().lower()
                if s in {"positive", "pos", "p", "bullish"}:
                    return "Positive"
                if s in {"negative", "neg", "n", "bearish"}:
                    return "Negative"
                if s in {"neutral", "neu", "0", "none"}:
                    return "Neutral"
                return "Unknown"

            if "sentiment_raw" in df.columns:
                df["sentiment"] = df["sentiment_raw"].apply(normalize_sentiment)
            else:
                # If sentiment_raw missing, add Unknown
                df["sentiment"] = "Unknown"

            # Layout: Two columns
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Latest News")

                if df.empty:
                    st.info("No news articles found for this ticker.")
                else:
                    filter_option = st.radio("Filter News", ["All", "Positive", "Negative", "Neutral", "Unknown"],
                                             horizontal=True)

                    if filter_option == "All":
                        filtered_df = df
                    else:
                        filtered_df = df[df["sentiment"] == filter_option]

                    if filtered_df.empty:
                        st.info("No articles match the selected filter.")
                    else:
                        for i, row in filtered_df.iterrows():
                            with st.container():
                                title = row.get("title") or "Untitled"
                                link = row.get("link") or ""
                                snippet = row.get("snippet") or ""
                                sentiment = row.get("sentiment", "Unknown")

                                st.markdown(f"### {title}")
                                if link:
                                    st.markdown(f"[Read more]({link})")
                                if snippet:
                                    st.markdown(snippet)
                                sentiment_color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
                                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>",
                                            unsafe_allow_html=True)
                                st.markdown("---")

            with col2:
                # Remove unsupported 'border=True' param (st.container has no 'border' arg)
                with st.container():
                    st.subheader("Sentiment")

                    # Guard: ensure sentiment column exists and is not empty
                    if "sentiment" not in df.columns or df["sentiment"].dropna().empty:
                        st.info("No sentiment data to show.")
                    else:
                        sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100
                        labels = sentiment_counts.index.tolist()
                        values = sentiment_counts.values.tolist()

                        # Donut chart
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
                        fig.update_layout(showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary
                        st.subheader("Sentiment Summary")
                        for label, value in zip(labels, values):
                            st.write(f"**{label}:** {value:.2f}%")

    with aiInsight:
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # ---------------- Calculate Parameters ---------------- #
                pe_ratio = info.get("forwardPE", info.get("trailingPE", 0))
                eps_growth = info.get("earningsQuarterlyGrowth", 0) * 100 if info.get(
                    "earningsQuarterlyGrowth") else 0
                beta = info.get("beta", 0)
                debt_to_equity = info.get("debtToEquity", 0)

                parameters = {
                    "P/E Ratio": (round(pe_ratio, 2) if pe_ratio else 0, 25),
                    "EPS Growth %": (round(eps_growth, 2), 15),
                    "Volatility (Beta)": (round(beta, 2), 1.0),
                    "Debt/Equity": (round(debt_to_equity, 2), 1.0),
                }

                # ---------------- Layout ---------------- #
                col1, col2 = st.columns([1, 1])

                # ---------------- LEFT: Bullet Charts ---------------- #
                with col1:
                    st.subheader(f"Key Parameters for {ticker}")

                    fig = make_subplots(
                        rows=len(parameters),
                        cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.08,
                        specs=[[{"type": "indicator"}] for _ in parameters]
                    )

                    row = 1
                    for param, (value, target) in parameters.items():
                        fig.add_trace(
                            go.Indicator(
                                mode="number+gauge",
                                value=value,
                                number={'prefix': f"{param}: ", 'font': {'size': 12}},
                                gauge={
                                    'shape': "bullet",
                                    'axis': {'range': [0, target * 1.5]},
                                    'bar': {'color': "royalblue"},
                                    'threshold': {
                                        'line': {'color': "red", 'width': 2},
                                        'thickness': 0.7,
                                        'value': target
                                    }
                                }
                            ),
                            row=row,
                            col=1
                        )
                        row += 1

                    fig.update_layout(
                        height=70 * len(parameters),
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_column_width=True)

                # ---------------- RIGHT: AI Insights ---------------- #
                with col2:
                    st.subheader("AI-Generated Insights")
                    time_option = st.selectbox("Select Time Horizon:", ["1 Year", "3 Year", "5 Year", "7 Year"])

                    if st.button("Generate Insights"):
                        prompt = f"""Analyze the stock {ticker} for a {time_option} horizon. 
                        Here are the parameters:

                        - P/E Ratio: {parameters['P/E Ratio'][0]} (Target: {parameters['P/E Ratio'][1]})
                        - EPS Growth %: {parameters['EPS Growth %'][0]} (Target: {parameters['EPS Growth %'][1]})
                        - Volatility (Beta): {parameters['Volatility (Beta)'][0]} (Target: {parameters['Volatility (Beta)'][1]})
                        - Debt/Equity: {parameters['Debt/Equity'][0]} (Target: {parameters['Debt/Equity'][1]})

                        Provide insights in bullet points.
                        """

                        insights = FetchPrice.get_response(prompt, history=[])
                        st.markdown("### 📌 Insights")
                        st.markdown(insights)

            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    main()
