import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_tweet(tweet):
    """Preprocess and clean the tweet."""
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = tweet.lower().strip()
    return tweet

def transform_text(text_list):
    """Convert text into TF-IDF feature representation."""
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix

if __name__ == "__main__":
    test_tweet = "Check out the new park in NYC! #citylife https://example.com @friend"
    print(clean_tweet(test_tweet))

    sample_tweets = ["I love programming", "I am so sad today", "What a beautiful day!", "I'm angry about this."]
    transformed_text = transform_text(sample_tweets)
    print("Text transformation completed.")
