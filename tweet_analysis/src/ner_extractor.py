import spacy

# Load spaCy's pre-trained model for NER
nlp = spacy.load("en_core_web_sm")

# Predefined list of cities
city_list = {
    "Bangalore", "Mysore", "Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad",
    "New York", "San Francisco", "London", "Paris", "Tokyo"
}

def extract_location(tweet):
    """Extract location from the tweet using Named Entity Recognition (NER) + City Lookup."""
    doc = nlp(tweet)
    locations = {ent.text for ent in doc.ents if ent.label_ == "GPE"}

    words = tweet.split()
    for word in words:
        if word in city_list:
            locations.add(word)

    return list(locations) if locations else []

if __name__ == "__main__":
    test_tweet = "There is a power outage in Mysore and flooding in Bangalore!"
    locations = extract_location(test_tweet)
    print("Extracted Locations:", locations)
