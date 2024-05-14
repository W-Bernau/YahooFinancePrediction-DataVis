import pandas as pd
from ntscraper import Nitter
import requests
import json

scraper = Nitter()

def get_tweets(hashtag, mode, size):
  tweets = scraper.get_tweets(hashtag, mode = mode, number = size)
  final_tweets = []

  for tweet in tweets['tweets']:
    data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
    final_tweets.append(data)

  data = pd.DataFrame(final_tweets, columns = ['link', 'text', 'date', 'Likes', 'Comments'])
  return data

##INSERT NAME OF SECURITY IN PLACE OF AMAZON
data = get_tweets("Amazon", "hashtag", 5)

def analyze_sentiment(text):
    url = "https://comprehend-it.p.rapidapi.com/predictions/ml-zero-nli-model"

    # Payload with the text to analyze and the labels for sentiment analysis
    payload = {
        "labels": ["positive", "negative", "neutral"],
        "text": text
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": '3b6bedaec6mshf3b79eff5c293d6p16085bjsnbec91df63ea3',  # Replace with your API key, refer to documentation for more info.
        "X-RapidAPI-Host": "comprehend-it.p.rapidapi.com"
    }

    # Make the API request
    response = requests.post(url=url, json= payload, headers= headers)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"
    
sentiments = []
for text in data['text']:
    result = analyze_sentiment(text)
    if result:
    # Formatting the sentiment scores as percentages
        sentiment_scores = result['outputs']
        sentiment_str = "; ".join([f"{sent}: {round(score * 100, 2)}%" for sent, score in sentiment_scores.items()])
    else:
        sentiment_str = 'Error'
    sentiments.append(sentiment_str)

# Add the sentiments to the DataFrame
data['Sentiment'] = sentiments
print(data['Sentiment'],data['text'])

#To-Do:
#Assign each tweet a value -.1, 0, .1 depending on its majority sentiment, then count them all to calculate a popular "rating"


