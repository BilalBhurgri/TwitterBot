import subprocess, requests, time, json, random, os, tweepy, argparse
from dotenv import load_dotenv
from unittest.mock import MagicMock
from bot.scrape import get_responses, get_responses_text
from try_models.query import query_and_generate
from try_models import query_full_paper_verbose
import boto3
from try_models.older_code.query import query_and_generate

parser = argparse.ArgumentParser(description='Generate tweets from paper database')
parser.add_argument('--folder_name', required=False, help='DB name', default="papers2")
parser.add_argument('--model_type', required=False, default="facebook/bart-large-cnn", help='Specify summarizer model. Default: facebook/bart-large-cnn')
parser.add_argument('--db_name', required=False, default='papers', help='The actual DB name within chroma.sqlite3. Default is "papers"')
parser.add_argument('--topic', default=None, help='Optional topic to focus on')
parser.add_argument('--num_papers', type=int, default=3, help='Number of papers to tweet about')
parser.add_argument('--days', type=int, default=None, help='Focus on papers from last N days')
parser.add_argument('--post', action='store_true', help='Actually post to Twitter')
args = parser.parse_args()

load_dotenv()

s3 = boto3.client('s3', region_name='us-west-1')

if args.post:
    # Twitter API credentials - store these in .env
    # Initialize Tweepy's Client using OAuth 1.0a credentials
    client = tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
    )
else:
    client = MagicMock()
    client.create_tweet.return_value.data = {"id": "1234567890"}

current_dir = os.path.dirname(os.path.abspath(__file__))

# def generate_response(prompt):
#     res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream":True})
#     print("Full response:", res.text)
#     return res.json()["response"].strip()

def post_tweet(tweet):
    try:
        # Post the tweet
        response = client.create_tweet(text=tweet)
        # Show the tweet ID or URL
        print("✅ Tweet posted! Tweet ID:", response.data["id"])
        print(f"https://twitter.com/user/status/{response.data['id']}")
    except Exception as e:
        print(f"❌ Failed to post tweet: {str(e)}")

def generate_tweet_from_full_paper():
    """
    Pick 3 random papers and generate tweets about them by using query_full_paper_verbose.py
    Returns: list of tuples (paper_id, tweet)
    """
    papers_response = s3.list_objects_v2(
        Bucket=os.environ.get('BUCKET_NAME'),
        Prefix='papers/'
    )

    papers_keys = [obj['Key'] for obj in papers_response.get('Contents', [])]

    for obj in papers_response.get('Contents', []):
        print(f"File: {obj['Key']}")
        print(f"Modified: {obj['LastModified']}")
        print("---")
        # Key format is papers/2412.00857.txt
        papers_keys.append(obj['Key'])
    
    # Pick 3 random papers
    selected_papers = random.sample(papers_keys, min(3, len(papers_keys)))
    summaries = []

    for paper_key in selected_papers:
        # Get the summary from the GPU server
        summary = requests.get(f'http://{os.environ.get("VM_GPU_INTERNAL_IP")}:5000/get-paper-summary/{paper_key}')
        summary = summary.json()['summary']
        summaries.append((paper_key, summary))

    return summaries

def generate_tweet_from_embeddings():
    """
    Generates tweets by using query.py, which feeds embeddings into a model
    Returns: list of tuples (paper_id, tweet)
    """
    tweets = query_and_generate(args)
    return tweets


def main():
    """
    This should be able to trigger 2 different kinds of tweet generations: 
    - Pick a random paper and generate a tweet about it by using query_full_paper_verbose.py
        - AKA feeding the full paper text to a model with a large context window 
    - Use query.py and generate a tweet by feeding embeddings to a model
    """
    full_paper_tweets = generate_tweet_from_full_paper()
    embeddings_tweets = generate_tweet_from_embeddings()

    # TODO: just generate stuff with full_paper_tweets and embedding_tweets 
    for paper_id, tweet in embeddings_tweets:
      return tweet
    
    for paper_id, tweet in full_paper_tweets:
      return tweet
        

if __name__ == "__main__":
    main()