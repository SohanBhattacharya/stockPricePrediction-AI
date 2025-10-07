import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import yfinance as yf
from PIL import Image
from plotly.subplots import make_subplots
from nlp import NewsAndSentimentAnalysis as nlp
from priceFetch import FetchPrice

import pandas as pd
import plotly.graph_objects as go

def main():
    if __name__ == "__main__":
        # ---------------------------------- App Icon and Image Start --------------------------------------
        st.set_page_config(page_title="WhiteDragon AI", layout="wide")

        # Path to your downloaded image
        image_path = "Dragon Image.png"
        # Load the image
        img = Image.open(image_path)
        # Create two columns: one for the image, one for the text
        col1, col2 = st.columns([0.4, 5])

        with col1:
            st.image(img, width=100, caption="", use_container_width=False)

        with col2:
            st.markdown(
                "<h2 style='margin: 0; display: flex; align-items: center;'>WhiteDragon AI</h2>",
                unsafe_allow_html=True,
            )

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
                    label=f"{ticker} (Current Price)",
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

        ticker = st.text_input("Enter Stock Ticker", "INFY")
        st.write("Example: AAPL, TSLA, INFY")
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
        overview, newsAndSentiment, aiInsight = st.tabs(["📈 Company Overview", "📃 News and Sentiment Analysis", "🔍 AI Insights"])

        with overview:
            if ticker:
                # Fetch stock data
                data = yf.download(ticker, period="6mo", interval="1d")

                if not data.empty:
                    latest_close = float(data["Close"].iloc[-1])
                    prev_close = float(data["Close"].iloc[-2])

                    # Create indicator + line chart
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

                    # Fix: ensure we’re passing a Series, not DataFrame
                    close_prices = data["Close"]
                    if isinstance(close_prices, pd.DataFrame):
                        close_prices = close_prices.iloc[:, 0]

                    # Add line chart
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

                    # Split layout into two columns
                    col1, col2 = st.columns([1.8, 1.2])  # wider for graph, narrower for overview

                    with col1:
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        with st.container(border=True):
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
        with newsAndSentiment:

            # Header

            st.title("📈 News & Sentiment Analysis")

            # Fetch News
            news_data = nlp.fetch_google_news(ticker)

            # Analyze Sentiments
            for article in news_data:
                article["sentiment"] = nlp.analyze_sentiment(article["title"])

            # Convert to DataFrame
            df = pd.DataFrame(news_data)

            # Layout: Two columns
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Latest News")
                filter_option = st.radio("Filter News", ["All", "Positive", "Negative", "Neutral"], horizontal=True)

                filtered_df = df if filter_option == "All" else df[df["sentiment"] == filter_option]

                for i, row in filtered_df.iterrows():
                    with st.container():
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"[Read more]({row['link']})")
                        sentiment_color = "green" if row["sentiment"] == "Positive" else "red" if row[
                                                                                                      "sentiment"] == "Negative" else "gray"
                        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{row['sentiment']}</span>",
                                    unsafe_allow_html=True)
                        st.markdown("---")

            with col2:
                with st.container(border=True):
                    st.subheader("Sentiment")

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

                        st.plotly_chart(fig, use_container_width=True)

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