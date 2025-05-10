import subprocess, requests, time, json, random, os, tweepy, argparse
from dotenv import load_dotenv
from unittest.mock import MagicMock
from bot.scrape import get_responses, get_responses_text
from try_models.query import query_and_generate

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

def main():
    tweets = query_and_generate(args)
    for paper_id, tweet in tweets:
        # print(f"Generated Tweet for {paper_id}:")
        # print(tweet)
        # print("\n" + "-"*50 + "\n")
        # print("Do you want to post this? (y/n): ", end="")
        # answer = input().strip().lower()
        return tweet
        # if answer == "y":
        #     post_tweet(tweet)
        # else:
        #     print("❌ Tweet canceled.")
        

if __name__ == "__main__":
    main()