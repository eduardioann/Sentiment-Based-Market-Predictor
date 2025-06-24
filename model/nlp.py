import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tweets_df = pd.read_csv("data/raw/stock_tweets.csv")
prices_df = pd.read_csv("data/raw/stock_yfinance_data.csv")

tweets_tesla = tweets_df[tweets_df['Stock Name'] == 'TSLA']
prices_tesla = prices_df[prices_df['Stock Name'] == 'TSLA']

def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

tweets_tesla['cleaned_text'] = tweets_tesla['Tweet'].apply(clean_tweet)

tweets_tesla['date'] = pd.to_datetime(tweets_tesla['Date']).dt.date
prices_tesla['date'] = pd.to_datetime(prices_tesla['Date']).dt.date

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

tweets_tesla['roberta_sentiment'] = tweets_tesla['cleaned_text'].apply(lambda x: nlp(x)[0]['label'])

sentiment_map = {'LABEL_2': 1, 'LABEL_0': -1, 'LABEL_1': 0}
tweets_tesla['sentiment_score'] = tweets_tesla['roberta_sentiment'].map(sentiment_map)

daily_sentiment = tweets_tesla.groupby(['date']).agg({
    'sentiment_score': ['mean', 'count']
})
daily_sentiment.columns = ['avg_sentiment', 'tweet_count']
daily_sentiment = daily_sentiment.reset_index()

daily_sentiment.loc[(daily_sentiment['avg_sentiment'] < -1) | (daily_sentiment['avg_sentiment'] > 1), 'avg_sentiment'] = 0

prices_tesla['price_change_pct'] = (prices_tesla['Close'] - prices_tesla['Open']) / prices_tesla['Open'] * 100
daily_stock = prices_tesla.groupby(['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']).agg({
    'price_change_pct': 'mean'
}).reset_index()

combined_tesla = pd.merge(daily_sentiment, daily_stock, on='date', how='inner')

combined_tesla = combined_tesla[(combined_tesla['avg_sentiment'] >= -1) & (combined_tesla['avg_sentiment'] <= 1)]

tweets_tesla.to_csv("tesla_tweets.csv", index=False)
prices_tesla.to_csv("tesla_prices.csv", index=False)
combined_tesla.to_csv("tesla_combined.csv", index=False)

print("Analiză completă! Datele corelate au fost salvate în 'tesla_combined.csv'.")

correlation = combined_tesla['avg_sentiment'].corr(combined_tesla['price_change_pct'])
print(f"Coeficient de corelație Pearson: {correlation:.4f}")
