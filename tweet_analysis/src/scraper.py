import requests
from bs4 import BeautifulSoup

# Set the URL of the Twitter page you want to scrape (you can change this to a hashtag or any public Twitter page)
url = "https://twitter.com/elonmusk"  # Example: Elon Musk's Twitter page
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Send the request
response = requests.get(url, headers=headers)

# Parse the response using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Example: Try to extract tweet content (this will need to be adjusted for the page structure)
tweets = soup.find_all('div', {'class': 'css-1dbjc4n'})

# Print the tweets
for tweet in tweets:
    print(tweet.text)
