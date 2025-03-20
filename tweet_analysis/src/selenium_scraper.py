from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import time

# Set up the Chrome WebDriver using webdriver-manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open the Twitter page (or any public Twitter profile)
url = "https://twitter.com/elonmusk"  # Example: Elon Musk's Twitter page
driver.get(url)

# Wait for the page to load (adjust the time if necessary)
time.sleep(5)

# Scroll down to load more tweets (You can adjust the scroll count as needed)
for _ in range(3):  # Scroll 3 times to load more tweets
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(3)

# Now, we extract tweets
tweets = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')

# Print the extracted tweets
for tweet in tweets:
    print(tweet.text)

# Close the browser
driver.quit()
