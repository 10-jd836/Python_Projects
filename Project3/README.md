# Article Sentiment

This script downloads an online article, extracts its summary, and uses TextBlob to classify the summary's sentiment as **Positive**, **Neutral**, or **Negative**.

## Run

Install the dependencies:

```bash
python -m pip install newspaper3k textblob
```

Set the article URL in `main1.py` (the `url` value is blank by default), then run:

```bash
python main1.py
```

The script prints one sentiment label.
