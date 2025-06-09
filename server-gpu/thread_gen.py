import json
import boto3
from dotenv import load_dotenv
import os
from botocore.exceptions import ClientError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import sys
import torch

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from try_models.summary_to_tweet import (
    generate_tweet_qwen
)

load_dotenv()
s3 = boto3.client('s3', region_name='us-west-1')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

def thread_gen():
    with open("data/all_threads.json", "r") as f:
        data = json.load(f)
    paper_ids = [item["paper"].split('/')[-1] for item in data["threads"]]
    for paper_id in paper_ids:
        try:
            path = f"downloads/thread_downloads/Qwen3-4B/bot1/{paper_id}.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    result = json.load(f)
                summary = result["summary"]
                sentences = summary.split(".")
                tweets = []
                for i in range(min([len(sentences), 8])):
                    print(f"Sentence: {sentences[i]}")
                    tweet = generate_tweet_qwen(sentences[i], tokenizer, model, bot_num=1)
                    real_tweet = tweet.split('\n')[0]
                    if i == 0:
                        real_tweet += f"\n Link: https://arxiv.org/abs/{paper_id}"
                    print(f"Tweet: {real_tweet}")
                    tweets.append(real_tweet)
                output = {"thread": tweets}
                with open(f'threads/Qwen3-4B/bot1/{paper_id}.json', 'w') as f:
                    json.dump(output, f, indent=2)
        except ClientError as e:
            print(f"Error getting object {paper_id}.txt: {e}")
            continue

def main():
    thread_gen()

if __name__ == "__main__":
    main()