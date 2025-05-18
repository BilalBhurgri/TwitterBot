import os
import argparse
import tweepy
from dotenv import load_dotenv
from unittest.mock import MagicMock
from bot.scrape import get_responses, get_responses_text
from try_models.query import query_and_generate


class TweetBot:
    def __init__(self):
        """
        Initialize the TweetBot class, inits client so you can post
        """
        self.args = self.parse_args()
        load_dotenv()
        self.client = self.init_twitter_client()

    def parse_args(self):
        """
        Parse the arguments
        """
        parser = argparse.ArgumentParser(description='Generate tweets from paper database')
        parser.add_argument('--folder_name', default="papers2", help='DB name')
        parser.add_argument('--model_type', default="facebook/bart-large-cnn", help='Specify summarizer model')
        parser.add_argument('--db_name', default='papers', help='The actual DB name within chroma.sqlite3')
        parser.add_argument('--topic', default=None, help='Optional topic to focus on')
        parser.add_argument('--num_papers', type=int, default=3, help='Number of papers to tweet about')
        parser.add_argument('--days', type=int, default=None, help='Focus on papers from last N days')
        parser.add_argument('--post', action='store_true', help='Actually post to Twitter')
        return parser.parse_args()

    def init_twitter_client(self):
        """
        Inits the twitter client
        """
        return tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
        )
    def post_tweet(self, tweet):
        """
        Posts a tweet to Twitter
        """
        try:
            response = self.client.create_tweet(text=tweet)
            tweet_id = response.data["id"]
            print("✅ Tweet posted! Tweet ID:", tweet_id)
            print(f"https://twitter.com/user/status/{tweet_id}")
        except Exception as e:
            print(f"❌ Failed to post tweet: {str(e)}")

    def run(self):
        """
        Runs the tweet bot, we won't use this function most likely because we're using the scheduler
        """
        tweets = query_and_generate(self.args)
        for paper_id, tweet in tweets:
            if self.args.post:
                self.post_tweet(tweet)
            return tweet
