import yfinance as yf
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re

class FetchPrice:
    @staticmethod
    def get_etf_price(ticker):
        etf = yf.Ticker(ticker)
        price = etf.history(period="1d")["Close"].iloc[-1]
        return price

    @staticmethod
    def get_yesterday_etf_close(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")  # last 2 trading days
            if len(hist) >= 2:
                return hist["Close"].iloc[-2]  # yesterday's close
            elif len(hist) == 1:
                return hist["Close"].iloc[0]  # only one day available
            else:
                return None
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    # Load tokens from .env file
    load_dotenv()

    # Define a function that attempts to get a response using two tokens
    @staticmethod
    def get_response(prompt, history):
        tokens = ["HF_TOKEN_ONE", "HF_TOKEN_TWO"]

        for token_key in tokens:
            try:
                client = InferenceClient(
                    provider="hf-inference",
                    api_key=os.environ[token_key],
                )

                response = client.chat.completions.create(
                    model="HuggingFaceTB/SmolLM3-3B",
                    messages=history + [{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content.strip()

                # Remove <think>...</think> sections if present
                clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                return clean_answer

            except Exception:
                # Try next token if this one fails
                continue

        return "⚠️ Error: Unable to fetch response from tokens."