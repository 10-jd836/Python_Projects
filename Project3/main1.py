from textblob import TextBlob
from newspaper import Article

url = ''
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary

blob = TextBlob(text)
sentiment = blob.sentiment.polarity # -1 to 1

threshold = 0.05

def get_sentiment_label(score):
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    return "Negative"

print(get_sentiment_label(sentiment))



