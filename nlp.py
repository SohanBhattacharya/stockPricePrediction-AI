import requests
import xml.etree.ElementTree as ET
from textblob import TextBlob


class NewsAndSentimentAnalysis:

    @staticmethod
    def fetch_google_news(ticker, num_articles=10):
        """
        Fetch Google News articles for a ticker using RSS feed.
        Sentiment is derived ONLY from the news title.
        """

        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={ticker}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        try:
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception as e:
            return [{
                "title": "Error fetching news",
                "summary": str(e),
                "link": "",
                "sentiment": "Neutral"
            }]

        articles = []
        items = root.findall(".//item")[:num_articles]

        for item in items:
            title = item.findtext("title", "No Title").strip()
            summary = item.findtext("description", "").strip()
            link = item.findtext("link", "").strip()

            # 🔹 SENTIMENT FROM TITLE ONLY
            sentiment = NewsAndSentimentAnalysis.analyze_sentiment(title)

            articles.append({
                "title": title,
                "summary": summary,
                "link": link,
                "sentiment": sentiment
            })

        if not articles:
            return [{
                "title": "No news found",
                "summary": "",
                "link": "",
                "sentiment": "Neutral"
            }]

        return articles

    @staticmethod
    def analyze_sentiment(title):
        """
        Analyze sentiment of a news headline using TextBlob.
        Returns: Positive / Negative / Neutral
        """
        if not title:
            return "Neutral"

        try:
            polarity = TextBlob(title).sentiment.polarity
        except Exception:
            return "Neutral"

        # Headline-optimized thresholds
        if polarity > 0.05:
            return "Positive"
        elif polarity < -0.05:
            return "Negative"
        else:
            return "Neutral"
