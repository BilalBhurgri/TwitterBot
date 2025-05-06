import subprocess, requests, time, json, random, os, tweepy
from dotenv import load_dotenv
from scrape import get_responses, get_responses_text

load_dotenv()
auth = tweepy.OAuth1UserHandler(
    os.getenv("TWITTER_API_KEY"),
    os.getenv("TWITTER_API_SECRET"),
    os.getenv("TWITTER_ACCESS_TOKEN"),
    os.getenv("TWITTER_ACCESS_SECRET")
)
api = tweepy.API(auth)
current_dir = os.path.dirname(os.path.abspath(__file__))

# def post_tweet(content):
#     api.update_status(content)
#     print("✅ Tweet posted!")

def generate_response(prompt):
    res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream":True})
    print("Full response:", res.text)
    return res.json()["response"].strip()

def generate_post(context):
    print("1")

def post_tweet(content):
    print("[MOCK POST]", content)
    # TODO: Replace this with real post using Twitter API or save it locally.


#replies = get_responses_text(username="RanjayKrishna" , post_index=6, headless=False)
#summary_prompt = "Summarize or respond to these tweets with a short reply:\n\n" + "\n".join(replies)
#print(summary_prompt)
json_file_path = os.path.join(current_dir, '1408.5882.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
generated_tweet = generate_post()
post_tweet(generated_tweet)
