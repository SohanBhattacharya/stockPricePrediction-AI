import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


class NewsAndSentimentAnalysis:

    @staticmethod
    def fetch_google_news(ticker, num_articles=10):
        """
        Fetch Google News articles for a ticker and return list of dicts with:
        title, summary, link, sentiment
        """

        url = f"https://news.google.com/search?q={ticker}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            return [{"title": "Error fetching news", "summary": str(e), "link": "", "sentiment": "Neutral"}]

        articles = []
        items = soup.select("article")[:num_articles]

        for item in items:
            # Extract title safely
            title_tag = item.select_one("h3")
            title = title_tag.text.strip() if title_tag else "No Title"

            # Extract short summary (if available)
            summary_tag = item.select_one(".HO8did")
            summary = summary_tag.text.strip() if summary_tag else ""

            # Extract link safely
            link_tag = item.find("a")
            link = ""
            if link_tag and link_tag.get("href"):
                href = link_tag["href"]
                if href.startswith("./"):     # standard Google News relative link
                    link = "https://news.google.com" + href[1:]
                elif href.startswith("http"):
                    link = href

            # Perform sentiment on title + summary together
            combined_text = f"{title}. {summary}".strip()
            sentiment = NewsAndSentimentAnalysis.analyze_sentiment(combined_text)

            articles.append({
                "title": title,
                "summary": summary,
                "link": link,
                "sentiment": sentiment
            })

        # If no articles found → return fallback structure
        if not articles:
            return [{
                "title": "No news found",
                "summary": "",
                "link": "",
                "sentiment": "Neutral"
            }]

        return articles

    @staticmethod
    def analyze_sentiment(text):
        """
        Analyze sentiment using TextBlob.
        Returns: Positive / Negative / Neutral
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
        except:
            return "Neutral"

        if polarity > 0.05:
            return "Positive"
        elif polarity < -0.05:
            return "Negative"
        else:
            return "Neutral"
