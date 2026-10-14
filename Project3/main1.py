from textblob import TextBlob
from newspaper import Article
url = 'https://geopolicraticus.substack.com/p/man-made-horrors-beyond-your-comprehension'
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary
print(text)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity 
print(sentiment)
