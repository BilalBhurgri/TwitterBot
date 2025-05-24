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

    # Save results to JSON file
    filename = f"twitter_responses_{results['original_tweet']['username']}_{results['original_tweet']['id']}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filename}")


def collect_training_data(usernames, posts_per_user=5, responses_per_post=20, output_file="twitter_training_data.json"):
    """
    Collect training data from multiple users for fine-tuning
    
    Args:
        usernames: List of Twitter usernames to scrape
        posts_per_user: Number of posts to collect from each user
        responses_per_post: Number of responses to collect per post
        output_file: Output JSON file name
    """
    all_data = []
    
    for username in usernames:
        print(f"\nCollecting data from @{username}...")
        
        for post_idx in range(posts_per_user):
            try:
                results = get_responses(
                    username=username,
                    post_index=post_idx,
                    max_responses=responses_per_post,
                    headless=True,
                    verbose=True
                )
                
                # Create training examples (original tweet + responses as summaries)
                training_example = {
                    "text": results['original_tweet']['content'],
                    "summary": " ".join([resp['content'] for resp in results['responses'][:5]]),  # Use first 5 responses
                    "source_user": username,
                    "tweet_id": results['original_tweet']['id'],
                    "tweet_url": results['original_tweet']['url'],
                    "all_responses": results['responses']
                }
                
                all_data.append(training_example)
                print(f"Collected post {post_idx + 1}/{posts_per_user} from @{username}")
                
            except Exception as e:
                print(f"Error collecting post {post_idx} from @{username}: {e}")
                continue
    
    # Save all training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining data saved to: {output_file}")
    print(f"Total examples collected: {len(all_data)}")
    
    return all_data


def load_usernames_from_file(filename="usernames.txt"):
    """
    Load usernames from a text file
    
    Args:
        filename: Path to the text file containing usernames
        
    Returns:
        list: List of usernames
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Split by commas and clean up whitespace
        usernames = [username.strip() for username in content.split(',') if username.strip()]
        
        print(f"Loaded {len(usernames)} usernames from {filename}")
        for i, username in enumerate(usernames, 1):
            print(f"  {i}. @{username}")
            
        return usernames
        
    except FileNotFoundError:
        print("file not found error!")
        return [] # ['geoffreyhinton', 'Yoshua_Bengio', 'AndrewYNg']
    except Exception as e:
        print("exception!")
        return [] # ['geoffreyhinton', 'Yoshua_Bengio', 'AndrewYNg']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Twitter Scraper')
    parser.add_argument('--mode', choices=['collect', 'example'], default='login',
                       help='Mode to run: collect training data, or run basic example')
    parser.add_argument('--usernames-file', default='usernames.txt',
                       help='Path to file containing usernames (default: usernames.txt)')
    parser.add_argument('--posts-per-user', type=int, default=5,
                       help='Number of posts to collect per user (default: 5)')
    parser.add_argument('--responses-per-post', type=int, default=20,
                       help='Number of responses to collect per post (default: 20)')
    parser.add_argument('--output-file', default='twitter_training_data.json',
                       help='Output file for training data (default: twitter_training_data.json)')

    login()
    usernames = load_usernames_from_file(args.usernames_file)
    collect_training_data(
        usernames=usernames,
        posts_per_user=args.posts_per_user,
        responses_per_post=args.responses_per_post,
        output_file=args.output_file
    )
