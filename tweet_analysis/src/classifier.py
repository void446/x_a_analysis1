from transformers import pipeline
from preprocessing import clean_tweet
from ner_extractor import extract_location
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Traditional machine learning model for emotion classification
def train_ml_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# Load deep learning-based emotion classification model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_emotion(tweet):
    """Predict the emotion of a tweet after cleaning and extract location if necessary."""
    cleaned_tweet = clean_tweet(tweet)
    emotion_result = classifier(cleaned_tweet)
    emotion = emotion_result[0]['label']
    confidence = emotion_result[0]['score']
    
    # Extract location if the tweet contains negative sentiment
    locations = []
    if emotion == 'NEGATIVE':
        locations = extract_location(tweet)
    
    return emotion, confidence, locations

# Machine learning training pipeline
def train_model_pipeline(tweets, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(tweets)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = train_ml_model(X_train, y_train)
    return model  # No printed messages

# Test classifier with location extraction
if __name__ == "__main__":
    input_tweet = input("Please enter a tweet: ")
    
    emotion, confidence, locations = predict_emotion(input_tweet)
    print(f"\nTweet: {input_tweet}")
    print(f"Emotion: {emotion} (Confidence: {confidence:.2f})")
    if locations:
        print(f"Extracted Locations: {locations}")
    else:
        print("No locations extracted.")

    # Simulated model training (silent execution)
    sample_tweets = ["I love programming", "I am so sad today", "What a beautiful day!", "I'm angry about this."]
    sample_labels = [0, 1, 2, 3]  # Example labels
    train_model_pipeline(sample_tweets, sample_labels)

    print("-" * 50)
