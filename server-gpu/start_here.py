"""
This generates today's tweets. 
1. It picks random paper texts from the bucket
2. Generates summaries 
3. Generates eval
4. Generates tweets
5. Returns them as json in /results/<model_name>/bot0, bot1, bot2... Using different personas. 
"""

import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import random 
import argparse

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import the paper processing functions
from try_models.query_full_paper_verbose import (
    load_paper,
    generate_summary,
    generate_summary_mistral,
    sum_eval
)

from try_models.summary_to_tweet import (
    generate_tweet_qwen
)

load_dotenv()
s3 = boto3.client('s3', region_name='us-west-1')
response = s3.list_buckets()

model = None
tokenizer = None
model_name = "Qwen/Qwen3-4B"  # Default model

NUM_BOTS = 6
NUM_INDICES = 10

def load_model():
    """Load the model and tokenizer. Currently both are based on Qwen3-4B."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully")

def process_paper_for_bot(paper_id: str, bot_num: int, num_summaries: int, eval=True):
    """
    Process a paper and generate a summary, tweet, and optionally an eval
    Params: 
    paper_id: paper_id is same as arxiv ID
    eval: bool. If True, eval happens.
    """
    # Check if model is loaded
    if model is None:
        load_model()

    print(f"Curr paper_id = {paper_id}")
    paper_text = ''
    try:
        obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=f'papers/{paper_id}.txt')
        paper_text = obj['Body'].read().decode('utf-8')
    except ClientError as e:
        print(f"Error getting object papers/{paper_id}.txt: {e}")
    
    print(f"generating summary with {len(paper_text)} summary")

    if eval:
        best_summary, evaluation = sum_eval(paper_text, tokenizer, model, model_name, num_summaries)
    else:
        if 'mistral' in model_name.lower():
            best_summary = generate_summary_mistral(paper_text, tokenizer, model)
        else:
            best_summary = generate_summary(paper_text, tokenizer, model)

    print(f"APP.PY: CALLING GENERATE TWEET")
    tweet = generate_tweet_qwen(summary, tokenizer, model, max_new_tokens=25, bot_num=bot_num)
    real_tweet = tweet.split('\n')[-1]
    real_tweet += f"\n Link: https://arxiv.org/abs/{paper_id}"

    # This should be created in main()
    result = {
        'status': 'success',
        'summary': summary,
        'evaluation': evaluation,
        'tweet': tweet,
        'real_tweet': real_tweet,
    }

    today = datetime.now().strftime("%Y-%m-%d")
    s3_key = f"results/{model_name}/bot{bot_num}/{today}/{paper_id}.json"
    if eval:
        s3_key = f"results-eval/{model_name}/bot{bot_num}/{today}/{paper_id}.json"
    print(f"s3_key = {s3_key}")
    
    try:
        s3.put_object(
            Bucket=os.environ.get('BUCKET_NAME'),
            Key=s3_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )
    except ClientError as e:
        print(f"Error putting object {s3_key} into s3 dir: {e}")

    print(f"Put {result} into s3 bucket :)")

def get_all_papers_from_bucket():
    """
    Returns all paper keys. This excludes the paper path and extension,
    so it only keeps the arxiv paper ID.
    """
    response = s3.list_objects_v2(
        Bucket=os.environ.get('BUCKET_NAME'),
        Prefix='papers/'
    )

    paper_keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            paper_keys.append(obj['Key'].split("/")[-1].split('.txt')[0])
    
    return paper_keys[1:]  # first one is nothing


def main():
    """
    This randomly selects NUM_INDICES for each bot. So the bots end up posting
    about different papers.
    """
    parser = argparse.ArgumentParser(description='Pick specific subset of papers or randomly pick them, generate summaries, evals, tweets.')
    parser.add_argument('--num_summaries', default=2, help='Number of summaries to generate for eval', type=int)
    parser.add_argument('--model_name', default="Qwen/Qwen3-4B")
    parser.add_argument('--no_eval', action='store_true', help='Turns off eval when this flag is added')
    args = parser.parse_args()
    
    load_model()
    paper_ids = get_all_papers_from_bucket()
    print(f"All paper_keys: {paper_ids}")
    selected_paper_ids = random.sample(range(len(paper_ids)), NUM_INDICES)

    papers_per_bot = []

    for i in range(): # NUM_BOTS
        papers_per_bot.append(random.sample(range(len(paper_ids)), NUM_INDICES))
    # print(f"Selected random paper_keys: {selected_paper_keys}")
    print(f"Random idxs per bot: {papers_per_bot}")

    eval = not args.no_eval
    for i in range(1): # NUM_BOTS
        for idx in papers_per_bot[i]:
            process_paper_for_bot(paper_ids[idx], i, args.num_summaries, eval)
            

if __name__ == '__main__':
    main()
