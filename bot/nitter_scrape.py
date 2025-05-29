import requests
import json
import time
import argparse
from bs4 import BeautifulSoup
import random

def get_random_nitter_instance():
    """
    Returns a random Nitter instance URL
    """
    nitter_instances = [
        "https://nitter.net",
        "https://nitter.1d4.us",
        "https://nitter.kavin.rocks",
        "https://nitter.unixfox.eu",
        "https://nitter.moomoo.me"
    ]
    return random.choice(nitter_instances)

def get_user_tweets(username, max_tweets=5):
    """
    Scrapes tweets from a user's profile using Nitter
    
    Args:
        username: Twitter username without @
        max_tweets: Maximum number of tweets to collect
        
    Returns:
        list: List of tweet dictionaries
    """
    base_url = get_random_nitter_instance()
    tweets = []
    
    try:
        # Get user's timeline
        response = requests.get(f"{base_url}/{username}", timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch tweets for @{username}")
            return tweets
            
        soup = BeautifulSoup(response.text, 'html.parser')
        tweet_elements = soup.find_all('div', class_='timeline-item')
        
        for tweet in tweet_elements[:max_tweets]:
            try:
                # Get tweet content
                content_element = tweet.find('div', class_='tweet-content')
                content = content_element.get_text(strip=True) if content_element else "No content"
                
                # Get tweet metrics
                metrics = {}
                stats = tweet.find_all('span', class_='tweet-stat')
                for stat in stats:
                    value = stat.find('span', class_='tweet-stat-value')
                    if value:
                        metric_type = stat.get('title', '').lower()
                        metric_value = value.get_text(strip=True)
                        metrics[metric_type] = metric_value
                
                # Get timestamp
                time_element = tweet.find('span', class_='tweet-date')
                timestamp = time_element.find('a')['title'] if time_element and time_element.find('a') else None
                
                # Get tweet ID from the link
                tweet_link = tweet.find('a', class_='tweet-link')
                tweet_id = tweet_link['href'].split('/')[-1] if tweet_link else None
                
                tweet_data = {
                    "id": tweet_id,
                    "username": username,
                    "content": content,
                    "timestamp": timestamp,
                    "metrics": metrics,
                    "url": f"https://twitter.com/{username}/status/{tweet_id}" if tweet_id else None
                }
                
                tweets.append(tweet_data)
                
            except Exception as e:
                print(f"Error processing tweet: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error fetching tweets for @{username}: {str(e)}")
    
    return tweets

def get_tweet_responses(username, tweet_id, max_responses=20):
    """
    Gets responses to a specific tweet using Nitter
    
    Args:
        username: Twitter username without @
        tweet_id: ID of the tweet to get responses for
        max_responses: Maximum number of responses to collect
        
    Returns:
        dict: Dictionary containing original tweet and responses
    """
    base_url = get_random_nitter_instance()
    responses = []
    
    try:
        # Get the tweet page
        response = requests.get(f"{base_url}/{username}/status/{tweet_id}", timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch tweet {tweet_id}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get original tweet
        original_tweet = soup.find('div', class_='thread-line')
        if not original_tweet:
            return None
            
        content_element = original_tweet.find('div', class_='tweet-content')
        original_content = content_element.get_text(strip=True) if content_element else "No content"
        
        # Get responses
        response_elements = soup.find_all('div', class_='thread-line')[1:max_responses+1]
        
        for response in response_elements:
            try:
                # Get response content
                content_element = response.find('div', class_='tweet-content')
                response_content = content_element.get_text(strip=True) if content_element else "No content"
                
                # Get responder's username
                username_element = response.find('a', class_='username')
                response_username = username_element.get_text(strip=True) if username_element else "Unknown"
                
                # Get responder's handle
                handle_element = response.find('a', class_='fullname')
                response_handle = handle_element.get_text(strip=True) if handle_element else "Unknown"
                
                # Get timestamp
                time_element = response.find('span', class_='tweet-date')
                timestamp = time_element.find('a')['title'] if time_element and time_element.find('a') else None
                
                # Get metrics
                metrics = {}
                stats = response.find_all('span', class_='tweet-stat')
                for stat in stats:
                    value = stat.find('span', class_='tweet-stat-value')
                    if value:
                        metric_type = stat.get('title', '').lower()
                        metric_value = value.get_text(strip=True)
                        metrics[metric_type] = metric_value
                
                response_data = {
                    "username": response_username,
                    "handle": response_handle,
                    "content": response_content,
                    "timestamp": timestamp,
                    "metrics": metrics
                }
                
                responses.append(response_data)
                
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                continue
        
        result = {
            "original_tweet": {
                "id": tweet_id,
                "username": username,
                "content": original_content,
                "url": f"https://twitter.com/{username}/status/{tweet_id}"
            },
            "responses": responses
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching responses for tweet {tweet_id}: {str(e)}")
        return None

def collect_training_data(usernames, posts_per_user=5, responses_per_post=20, output_file="twitter_training_data.json"):
    """
    Collect training data from multiple users
    
    Args:
        usernames: List of Twitter usernames to scrape
        posts_per_user: Number of posts to collect from each user
        responses_per_post: Number of responses to collect per post
        output_file: Output JSON file name
    """
    all_data = []
    
    for username in usernames:
        print(f"\nCollecting data from @{username}...")
        
        # Get user's tweets
        tweets = get_user_tweets(username, max_tweets=posts_per_user)
        
        for tweet in tweets:
            try:
                # Get responses for this tweet
                results = get_tweet_responses(
                    username=username,
                    tweet_id=tweet['id'],
                    max_responses=responses_per_post
                )
                
                if results:
                    # Create training example
                    training_example = {
                        "text": results['original_tweet']['content'],
                        "summary": " ".join([resp['content'] for resp in results['responses'][:5]]),  # Use first 5 responses
                        "source_user": username,
                        "tweet_id": results['original_tweet']['id'],
                        "tweet_url": results['original_tweet']['url'],
                        "all_responses": results['responses']
                    }
                    
                    all_data.append(training_example)
                    print(f"Collected responses for tweet {tweet['id']}")
                
                # Be nice to the servers
                time.sleep(2)
                
            except Exception as e:
                print(f"Error collecting responses for tweet {tweet['id']}: {e}")
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
        print(f"File {filename} not found!")
        return []
    except Exception as e:
        print(f"Error loading usernames: {str(e)}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Twitter Scraper using Nitter')
    parser.add_argument('--usernames-file', default='usernames.txt',
                       help='Path to file containing usernames (default: usernames.txt)')
    parser.add_argument('--posts-per-user', type=int, default=5,
                       help='Number of posts to collect per user (default: 5)')
    parser.add_argument('--responses-per-post', type=int, default=20,
                       help='Number of responses to collect per post (default: 20)')
    parser.add_argument('--output-file', default='twitter_training_data.json',
                       help='Output file for training data (default: twitter_training_data.json)')

    args = parser.parse_args()
    
    usernames = load_usernames_from_file(args.usernames_file)
    if usernames:
        collect_training_data(
            usernames=usernames,
            posts_per_user=args.posts_per_user,
            responses_per_post=args.responses_per_post,
            output_file=args.output_file
        ) 