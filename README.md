
---

# 📈 Stock Price Prediction AI

**An AI-powered financial dashboard that blends real-time stock data, market sentiment, and visualization for smarter investing insights.**


---

---

## 🚀 Overview

**Stock Price Prediction AI** is an interactive **Streamlit web app** that combines **machine learning**, **financial data**, and **natural language processing (NLP)** to help users explore market movements, investor sentiment, and price trends — all in one clean dashboard.

The app fetches **real-time stock data** from Yahoo Finance, performs **sentiment analysis** on recent financial news, and visualizes insights using Plotly — empowering investors to make data-driven decisions.

---

## 🧠 Key Features

✅ **Live Stock Data** — Real-time price, volume, and trend analysis via `yfinance`
✅ **Sentiment Analysis** — Financial news sentiment using **TextBlob** + **Hugging Face models**
✅ **Interactive Visuals** — Plotly charts with Streamlit metric cards
✅ **Web Scraping** — Get live news headlines with **BeautifulSoup**
✅ **Secure Tokens** — Environment management via **python-dotenv**
✅ **Modular Code** — Organized architecture for easy customization

* `priceFetch.py` → stock price retrieval
* `nlp.py` → sentiment and news analysis
* `mainPage.py` → Streamlit dashboard

---

## 🏗️ Tech Stack

| Category            | Tools                       |
| ------------------- | --------------------------- |
| **Frontend / UI**   | Streamlit, streamlit-extras |
| **Data & Analysis** | Pandas, Plotly, TextBlob    |
| **APIs & ML**       | yfinance, Hugging Face Hub  |
| **Web Scraping**    | Requests, BeautifulSoup     |
| **Configuration**   | python-dotenv               |
| **Language**        | Python 3.9+                 |

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/SohanBhattacharya/stockPricePrediction-AI.git
cd stockPricePrediction-AI
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables

Create a `.env` file in the project root and add your tokens:

```bash
HF_TOKEN_ONE=your_huggingface_token_1
HF_TOKEN_TWO=your_huggingface_token_2
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run mainPage.py
```

---

## 📊 Example Outputs

* 📈 **Stock Trend Visualization** — Interactive line and candlestick charts
* 🗞️ **Sentiment Summary** — Positive/Negative sentiment percentages
* 🧮 **Model Fit Demo** — Visuals showing AI model fitting over time

---

## 📦 Requirements

```
streamlit
streamlit-extras
yfinance
pillow
plotly
pandas
requests
beautifulsoup4
textblob
huggingface_hub
python-dotenv
```

---

## 💡 Future Enhancements

✨ Add predictive models (LSTM, Prophet, or regression-based forecasting)
✨ Portfolio tracking & backtesting features
✨ Sentiment trends across multiple tickers
✨ Multi-language news sentiment analysis

---

## 🧾 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions are welcome!
To propose changes:

1. Fork the repository
2. Create a new branch
3. Submit a pull request with your improvements

---

## 🌐 Author

👤 **Sohan Bhattacharya**
🔗 [GitHub](https://github.com/SohanBhattacharya)
💼 *Developer & Data Enthusiast | Building AI-powered financial tools*

---
