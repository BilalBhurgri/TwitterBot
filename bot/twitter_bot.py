import subprocess
import requests
import time
import json
import random
import os
import random
import tweepy
import argparse
import datetime
from datetime import date
from dotenv import load_dotenv
from unittest.mock import MagicMock
from bot.scrape import get_responses, get_responses_text
from try_models.query import query_and_generate
from try_models import query_full_paper_verbose
import boto3
from try_models.older_code.query import query_and_generate
from zoneinfo import ZoneInfo

"""
Twitterbot object
Used in scheduler to schedule per bot to allow bot to post and generate tweets (through gpu server)
"""

LOCAL_TZ = ZoneInfo("America/Los_Angeles")

class TwitterBot:

    """
    Initialization function for twitter bot
        Arguments:
            account - Integer representation of which twitter account bot is associated with
            folder_name - name of DB directory
            model_type - summarizer model
            db_name - The actual name of DB
            topic - topic to post about
            num_papers - number of papers to tweet about
            days - Focus on papers from last N days
            post - Boolean value of whether to post or not
    """
    def __init__(
        self,
        account,
        folder_name="papers2",
        model_type="facebook/bart-large-cnn",
        db_name="papers",
        topic=None,
        num_papers=3,
        days=None,
        post=False
    ):
        self.folder_name = folder_name
        self.model_type = model_type
        self.db_name = db_name
        self.topic = topic
        self.num_papers = num_papers
        self.days = days
        self.post = post
        self.account = account
        load_dotenv()
        self.s3 = boto3.client('s3', region_name='us-west-1')
        self.summaries = set()
        self.summariesUsed = set()
        self.prefix = f'results/Qwen/Qwen3-4B/bot{self.account}/'
        self.get_summaries()

        if self.post:
            env_generic = ["TWITTER_API_KEY", "TWITTER_API_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"]
            # append account number to get key
            env_keys = [key + str(self.account) for key in env_generic]
            self.client = tweepy.Client(
                consumer_key=os.getenv(env_keys[0]),
                consumer_secret=os.getenv(env_keys[1]),
                access_token=os.getenv(env_keys[2]),
                access_token_secret=os.getenv(env_keys[3])
            )
 
            try:
                # Get the authenticated user's info
                user = self.client.get_me()
                print(f"✅ Auth successful. Logged in as @{user.data.username}")
            except tweepy.TweepyException as e:
                print(f"❌ Auth failed: {e}")
        else:
            self.client = MagicMock()
            self.client.create_tweet.return_value.data = {"id": "1234567890"}

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def post_tweet(self, tweet):
        try:
            response = self.client.create_tweet(text=tweet)
            print("✅ Tweet posted! Tweet:", tweet)
            self.summariesUsed.add(tweet)
        except Exception as e:
            print(f"❌ Failed to post tweet: {str(e)}")

    def generate_tweet_from_full_paper(self):
        papers_response = self.s3.list_objects_v2(
            Bucket=os.environ.get('BUCKET_NAME'),
            Prefix='papers/'
        )
        
        papers_keys = []
        for obj in papers_response.get('Contents', []):
            print(f"File: {obj['Key']}")
            print(f"Modified: {obj['LastModified']}")
            print("---")
            papers_keys.append(obj['Key'])

        selected_papers = random.sample(papers_keys, min(3, len(papers_keys)))
        summaries = []

        for paper_key in selected_papers:
            summary_response = requests.get(f'http://{os.environ.get("VM_GPU_INTERNAL_IP")}:5000/get-paper-summary/{paper_key}')
            summary = summary_response.json()['summary']
            summaries.append((paper_key, summary))

        return summaries

    def get_summaries(self):
        formatted_date = (datetime.datetime.now(LOCAL_TZ)).strftime("%Y-%m-%d")
        print(f"Retrieving FROM {self.prefix}{formatted_date}/")
        papers_response = self.s3.list_objects_v2(
            Bucket=os.environ.get('BUCKET_NAME'),
            Prefix=f'{self.prefix}{formatted_date}/'
        )
        for obj in papers_response['Contents']:
            key = obj['Key']
            content = self.s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=key)['Body'].read().decode('utf-8')
            if content not in self.summariesUsed:
                contentJson = json.loads(content)
                print(contentJson.get("real_tweet"))
                self.summaries.add(contentJson.get("real_tweet"))
    
    def get_random_summary(self):
        random_summary = random.choice(list(self.summaries))
        self.summaries.remove(random_summary)
        return random.choice(list(self.summaries))


    def generate_tweet_from_embeddings(self):
        return query_and_generate(self)

    # Won't utilize because scheduler class must confirm before posting
    def run(self):
        self.get_summaries()
        return self.get_random_summary()
