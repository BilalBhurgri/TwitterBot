"""
This uses twitter API v2 free version to retrieve tweets.
"""
import tweepy
import json
import time
import argparse
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

class TwitterScraperV2:
    def __init__(self):
        # Initialize Twitter API v2 client
        # bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        # api_key = os.getenv("TWITTER_API_KEY")
        # api_secret = os.getenv("TWITTER_API_SECRET")
        # access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        # access_secret = os.getenv("TWITTER_ACCESS_SECRET")

        # print(f"Bearer Token: {'✓' if bearer_token else '✗'} ({'*' * 10 + bearer_token[-4:] if bearer_token else 'Missing'})")
        # print(f"API Key: {'✓' if api_key else '✗'} ({'*' * 10 + api_key[-4:] if api_key else 'Missing'})")
        # print(f"API Secret: {'✓' if api_secret else '✗'} ({'*' * 10 + api_secret[-4:] if api_secret else 'Missing'})")
        # print(f"Access Token: {'✓' if access_token else '✗'} ({'*' * 10 + access_token[-4:] if access_token else 'Missing'})")
        # print(f"Access Secret: {'✓' if access_secret else '✗'} ({'*' * 10 + access_secret[-4:] if access_secret else 'Missing'})")

        self.client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            # consumer_key=os.getenv("TWITTER_API_KEY"),
            # consumer_secret=os.getenv("TWITTER_API_SECRET"),
            # access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            # access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
            wait_on_rate_limit=True
        )

    def get_user_tweets(self, username, max_tweets=10, days_ago=None):
        """
        Get tweets from a user's timeline
        
        Args:
            username: Twitter username without @
            max_tweets: Maximum number of tweets to collect
            days_ago: Only get tweets from the last N days
            
        Returns:
            list: List of tweet dictionaries
        """
        try:
            # Get user ID from username
            user = self.client.get_user(username=username)
            if not user.data:
                print(f"User @{username} not found")
                return []
            
            user_id = user.data.id
            
            # Set up time filter if specified
            start_time = None
            if days_ago:
                start_time = datetime.utcnow() - timedelta(days=days_ago)
            
            # Get tweets
            tweets = []
            pagination_token = None
            
            while len(tweets) < max_tweets:
                # Ensure max_results is at least 10
                current_max = max(10, min(100, max_tweets - len(tweets)))
                
                response = self.client.get_users_tweets(
                    user_id,
                    max_results=current_max,
                    start_time=start_time,
                    tweet_fields=['created_at', 'public_metrics', 'conversation_id'],
                    pagination_token=pagination_token
                )
                
                if not response.data:
                    break
                    
                for tweet in response.data:
                    tweet_data = {
                        "id": str(tweet.id),
                        "username": username,
                        "content": tweet.text,
                        "timestamp": tweet.created_at.isoformat(),
                        "metrics": tweet.public_metrics,
                        "conversation_id": str(tweet.conversation_id),
                        "url": f"https://twitter.com/{username}/status/{tweet.id}"
                    }
                    tweets.append(tweet_data)
                
                if not response.meta.get('next_token'):
                    break
                    
                pagination_token = response.meta['next_token']
                time.sleep(1)  # Rate limiting
            
            return tweets[:max_tweets]  # Ensure we don't return more than requested
            
        except Exception as e:
            print(f"Error fetching tweets for @{username}: {str(e)}")
            return []

    def get_tweet_responses(self, username, tweet_id, max_responses=20):
        """
        Get responses to a specific tweet
        
        Args:
            username: Twitter username without @
            tweet_id: ID of the tweet to get responses for
            max_responses: Maximum number of responses to collect
            
        Returns:
            dict: Dictionary containing original tweet and responses
        """
        try:
            # Get original tweet
            tweet = self.client.get_tweet(
                tweet_id,
                tweet_fields=['created_at', 'public_metrics', 'conversation_id']
            )
            
            if not tweet.data:
                print(f"Tweet {tweet_id} not found")
                return None
            
            # Get responses
            responses = []
            pagination_token = None
            
            while len(responses) < max_responses:
                # Ensure max_results is at least 10
                current_max = max(10, min(100, max_responses - len(responses)))
                
                response = self.client.search_recent_tweets(
                    query=f"conversation_id:{tweet.data.conversation_id}",
                    max_results=current_max,
                    tweet_fields=['created_at', 'public_metrics', 'author_id'],
                    user_fields=['username', 'name'],
                    expansions=['author_id'],
                    pagination_token=pagination_token
                )
                
                if not response.data:
                    break
                
                # Create user lookup dictionary
                users = {user.id: user for user in response.includes['users']}
                
                for tweet in response.data:
                    if str(tweet.id) == str(tweet_id):  # Skip original tweet
                        continue
                        
                    user = users[tweet.author_id]
                    response_data = {
                        "username": user.name,
                        "handle": f"@{user.username}",
                        "content": tweet.text,
                        "timestamp": tweet.created_at.isoformat(),
                        "metrics": tweet.public_metrics
                    }
                    responses.append(response_data)
                
                if not response.meta.get('next_token'):
                    break
                    
                pagination_token = response.meta['next_token']
                time.sleep(1)  # Rate limiting
            
            result = {
                "original_tweet": {
                    "id": str(tweet_id),
                    "username": username,
                    "content": tweet.data.text,
                    "url": f"https://twitter.com/{username}/status/{tweet_id}"
                },
                "responses": responses[:max_responses]  # Ensure we don't return more than requested
            }
            
            return result
            
        except Exception as e:
            print(f"Error fetching responses for tweet {tweet_id}: {str(e)}")
            return None

    def collect_training_data(self, usernames, posts_per_user=5, responses_per_post=20, output_file="twitter_training_data.json"):
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
            tweets = self.get_user_tweets(username, max_tweets=posts_per_user)
            
            for tweet in tweets:
                try:
                    # Get responses for this tweet
                    results = self.get_tweet_responses(
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
                    
                    time.sleep(2)  # Rate limiting
                    
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
    parser = argparse.ArgumentParser(description='Twitter Scraper using API v2')
    parser.add_argument('--usernames-file', default='usernames.txt',
                       help='Path to file containing usernames (default: usernames.txt)')
    parser.add_argument('--posts-per-user', type=int, default=5,
                       help='Number of posts to collect per user (default: 5)')
    parser.add_argument('--responses-per-post', type=int, default=20,
                       help='Number of responses to collect per post (default: 20)')
    parser.add_argument('--output-file', default='twitter_training_data.json',
                       help='Output file for training data (default: twitter_training_data.json)')
    parser.add_argument('--days', type=int, default=None,
                       help='Only collect tweets from the last N days')

    args = parser.parse_args()
    
    scraper = TwitterScraperV2()
    usernames = load_usernames_from_file(args.usernames_file)
    if usernames:
        scraper.collect_training_data(
            usernames=usernames,
            posts_per_user=args.posts_per_user,
            responses_per_post=args.responses_per_post,
            output_file=args.output_file
        )

