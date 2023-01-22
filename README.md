# Sentiment analysis of show reviews

## Why?

- **Handmade**: I've [parsed](https://github.com/Extremesarova/shows_parsing) the dataset by myself
- **Big**, **imbalanced**, **multi-class** problem: `206 737` objects; 72% positive, 15% neutral and 12% negative reviews.
- **Long** texts: more than 512 tokens, so I need to come up with workarounds for transformer models for Russian language to beat the baseline created with classic ML approach.

## Solutions

### [Baseline: TF-IDF + Logistic Regression](https://github.com/Extremesarova/shows_sentiment_analysis/blob/main/notebooks/01_baseline.ipynb)

As a baseline, I've decided to choose a simple combination of TF-IDF for text vectorization and Logistic Regression for classification.  

The macro F1 score is `0.62` for baseline.

### Pretrained models

#### [HuggingFace](https://github.com/Extremesarova/shows_sentiment_analysis/blob/main/notebooks/02_pretrained_huggingface.ipynb)

#### [BERT Classifier as Feature Extractor](https://github.com/Extremesarova/shows_sentiment_analysis/blob/main/notebooks/03_bert-as-feature-extractor.ipynb)
