from playwright.sync_api import sync_playwright
import time
import json
import argparse

"""
Scraping methods are contained within this file they are developed with playwright

To use this you need to call python scrape.py and login manually and then press enter in terminal from
that point forward you should be able to use the methods because login state will be saved

Sample usage of get_responses is shown underneath the method
"""


def login():
    """
    Createst twitter_state.json enabling later logins
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://twitter.com/login")
        input("Log in manually, then press Enter...")
        context.storage_state(path="twitter_state.json")

def get_responses(username, post_index=0, max_responses=20, headless=False, verbose=False):
    """
    Scrapes responses to a specific post from an X user's timeline
    
    Args:
        username: The twitter username (without @)
        post_index: Which post to get (0 indexed)
        max_responses: Maximum number of responses to collect
        headless: Whether to run the browser in headless mode
        verbose: If true method prints information
    Returns:
        list: A list of dictionaries containing response data
            Schema of result
            list = {
            "original_tweet": {
                "id": tweet_id,
                "username": username,
                "content": original_content,
                "url": f"https://twitter.com/{username}/status/{tweet_id}"
            },
            "responses": responses[:max_responses]
                responses.append({
                    "username": response_username,
                    "handle": response_handle,
                    "content": response_content,
                    "timestamp": timestamp,
                    "metrics": metrics
                })
        }
    """
    # Input validation
    if not username:
        raise ValueError("Username is required")
    if post_index < 0:
        raise ValueError("Post index must be non-negative")
    if max_responses < 1:
        raise ValueError("Max responses must be positive")
    
    if verbose:
        print(f"Launching browser to fetch responses for @{username}'s post #{post_index}...")
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state="twitter_state.json", viewport={"width": 1280, "height": 800})
        page = context.new_page()
        
        # Go to page
        page.goto(f"https://twitter.com/{username}")
        
        # Wait for timeline to load
        page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
        
        # Get all tweets near top of page
        scrolls = 0
        # scrolls down further
        max_scrolls = 3
        tweets = []

        while len(tweets) <= post_index and scrolls < max_scrolls:
            page.mouse.wheel(0, 3000)
            time.sleep(2)
            tweets = page.query_selector_all('article[data-testid="tweet"]')
            scrolls += 1

        if post_index >= len(tweets):
            browser.close()
            raise IndexError(f"Post index {post_index} is out of range. User has only {len(tweets)} visible posts.")
        
        # Select the target tweet
        target_tweet = tweets[post_index]
        
        # Extract tweet ID from the permalink
        tweet_link = target_tweet.query_selector('a[href*="/status/"]')
        if not tweet_link:
            browser.close()
            raise Exception("Could not find tweet permalink")
        
        tweet_url = tweet_link.get_attribute('href')
        tweet_id = tweet_url.split('/status/')[1].split('/')[0]
        if verbose:
            print(f"Found tweet with ID: {tweet_id}")
        
        # Navigate to the tweet's page to view responses
        page.goto(f"https://twitter.com/{username}/status/{tweet_id}")
        
        # Wait for responses to load
        time.sleep(3)
        page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
        
        # Get the first (original) tweet content
        original_tweet = page.query_selector('article[data-testid="tweet"]')
        original_tweet_text = original_tweet.query_selector('div[data-testid="tweetText"]')
        original_content = original_tweet_text.inner_text() if original_tweet_text else "No text content"
        if verbose:
            print(f"Original tweet: {original_content[:50]}...")
        
        # Initialize responses list
        responses = []
        
        # Get all response tweets
        last_response_count = 0
        
        # Scroll and collect responses until we have enough
        while len(responses) < max_responses:
            # Find all responses (excluding the original tweet)
            response_tweets = page.query_selector_all('article[data-testid="tweet"]')
            
            # Skip the first one which is the original tweet
            response_tweets = response_tweets[1:]
            
            # Process new responses
            for i in range(last_response_count, len(response_tweets)):
                if len(responses) >= max_responses:
                    break
                    
                tweet = response_tweets[i]
                
                # Extract username
                user_element = tweet.query_selector('div[data-testid="User-Name"]')
                username_element = user_element.query_selector('span')
                response_username = username_element.inner_text() if username_element else "Unknown"
                
                # # Extract handle
                # handle_element = user_element.query_selector('span:has-text("@")')
                # response_handle = handle_element.inner_text() if handle_element else "Unknown"
                # This is safer than using `span:has-text("@")`
                handle_element = user_element.query_selector_all('span')
                response_handle = next((s.inner_text() for s in handle_element if s.inner_text().startswith('@')), "Unknown")

                # Extract content
                content_element = tweet.query_selector('div[data-testid="tweetText"]')
                response_content = content_element.inner_text() if content_element else "No text content"
                
                # Extract metrics (likes, replies, retweets)
                metrics = {}
                metrics_elements = tweet.query_selector_all('div[data-testid$="-count"]')
                for metric_el in metrics_elements:
                    metric_type = metric_el.get_attribute('data-testid').replace('-count', '')
                    metric_value = metric_el.inner_text()
                    metrics[metric_type] = metric_value
                
                # Extract timestamp
                time_element = tweet.query_selector('time')
                timestamp = time_element.get_attribute('datetime') if time_element else None
                
                # Add response to list
                responses.append({
                    "username": response_username,
                    "handle": response_handle,
                    "content": response_content,
                    "timestamp": timestamp,
                    "metrics": metrics
                })
                if verbose:
                    print(f"Collected response {len(responses)}/{max_responses} from {response_handle}")
            
            # May have reached end of response
            if len(response_tweets) == last_response_count:
                if verbose:
                    print("No new responses found after scrolling. May have reached the end.")
                break
                
            last_response_count = len(response_tweets)
            
            # Scroll down to load more responses
            page.mouse.wheel(0, 10000)
            time.sleep(2)  

        browser.close()
        result = {
            "original_tweet": {
                "id": tweet_id,
                "username": username,
                "content": original_content,
                "url": f"https://twitter.com/{username}/status/{tweet_id}"
            },
            "responses": responses[:max_responses]
        }
        
        return result

def get_responses_text(username, post_index=0, max_responses=20, headless=True, verbose=False):
    responses_text = []
    result = get_responses(username, post_index, max_responses, headless, verbose)
    for response in result["responses"]:
        responses_text.append(response["content"])
    return responses_text

def basic_example():
    # Gets the 10 most recent responses to elon's latest tweet
    results = get_responses(
        username="elonmusk",  # X username without the @ symbol
        post_index=2, # 0 indexed
        max_responses=10,
        headless=False
    )
    
    # Print the original tweet
    print(f"Original tweet: {results['original_tweet']['content'][:100]}...")
    print(f"Tweet URL: {results['original_tweet']['url']}")
    print(f"Found {len(results['responses'])} responses")
    
    for i, response in enumerate(results['responses'][:3]):
        print(f"\nResponse #{i+1} from {response['handle']}:")
        print(f"Content: {response['content'][:100]}...")


if __name__ == "__main__":
    login()