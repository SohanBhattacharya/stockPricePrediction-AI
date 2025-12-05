import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


class NewsAndSentimentAnalysis:
    # Function to fetch Google News articles for a ticker
    def fetch_google_news(ticker, num_articles=10):
        url = f"https://news.google.com/search?q={ticker}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        for item in soup.select("article")[:num_articles]:
            title = item.text
            link_tag = item.find("a")
            link = "https://news.google.com" + link_tag["href"][1:] if link_tag else ""
            articles.append({"title": title, "link": link})
        return articles

    # Function to perform sentiment analysis on a given news
    def analyze_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"


