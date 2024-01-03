import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline

## Transformers pipeline test by Sentiment analysis
sentiment_classifier = pipeline("sentiment-analysis")
sentiment = sentiment_classifier(["I've been waiting for a HuggingFace course my whole life.",
                        "I hate this so much!"])

print(sentiment)

## Zero-shot Classification
zs_classifier = pipeline("zero-shot-classification")
zs = zs_classifier("This is a course about the Transformers library",
                   candidate_labels=["education","politics","business"],)
print(zs)

## Text Generation
text_generator = pipeline("text-generation")
# text_generator = pipeline("text-generation", model="distilgpt2") # Load specific model from HuggingFace

