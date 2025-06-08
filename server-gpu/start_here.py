"""
This generates today's tweets. 
1. It picks random paper texts from the bucket
2. Generates summaries 
3. Generates eval
4. Generates tweets
5. Returns them as json in /results/<model_name>/bot0, bot1, bot2... Using different personas. 
6. Tracks which bots processed which papers
"""

import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    generate_tweet_qwen,
    generate_tweet_mistral,
    generate_tweet_olmo,
    generate_tweet_llama,
    summary_splitting
)

load_dotenv()
s3 = boto3.client('s3', region_name='us-west-1')
response = s3.list_buckets()

model = None
tokenizer = None
model_name = "Qwen/Qwen3-4B"

NUM_BOTS = 6

# 8-bit config with more options
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Default threshold
)

def load_model():
    """
    Fixes the model and model as Qwen/Qwen3-4B.
    The model for summary->tweet can vary.
    """
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if '7b' or '8b' in model_name.lower():
        print('running with 8bit!!')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        # assumes smaller model
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully")
    

def load_processed_papers_tracker():
    """Load the tracking file from S3. Returns empty dict if file doesn't exist."""
    tracking_key = "tracking/processed_papers.json"
    
    try:
        obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=tracking_key)
        tracking_data = json.loads(obj['Body'].read().decode('utf-8'))
        print(f"Loaded existing tracking data with {len(tracking_data.get('processed_papers', []))} papers")
        return tracking_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print("No existing tracking file found. Creating new one.")
            return {
                "processed_papers": [],
                "papers_by_bot": {str(i): [] for i in range(NUM_BOTS)},  # Track which papers each bot processed
                "bots_by_paper": {},  # Track which bots processed each paper
                "last_updated": datetime.now().isoformat(),
                "total_processed": 0
            }
        else:
            print(f"Error loading tracking file: {e}")
            return {
                "processed_papers": [],
                "papers_by_bot": {str(i): [] for i in range(NUM_BOTS)},
                "bots_by_paper": {},
                "last_updated": datetime.now().isoformat(),
                "total_processed": 0
            }

def save_processed_papers_tracker(tracking_data):
    """Save the tracking file to S3."""
    tracking_key = "tracking/processed_papers.json"
    tracking_data["last_updated"] = datetime.now().isoformat()
    tracking_data["total_processed"] = len(tracking_data["processed_papers"])
    
    try:
        s3.put_object(
            Bucket=os.environ.get('BUCKET_NAME'),
            Key=tracking_key,
            Body=json.dumps(tracking_data, indent=2),
            ContentType='application/json'
        )
        print(f"Updated tracking file with {tracking_data['total_processed']} total papers")
    except ClientError as e:
        print(f"Error saving tracking file: {e}")

def is_paper_already_processed(paper_id, bot_num, tracking_data, eval_mode=False):
    """Check if a paper has already been processed by a specific bot."""
    for entry in tracking_data["processed_papers"]:
        if (entry["paper_id"] == paper_id and 
            entry["bot_num"] == bot_num and 
            entry["eval_mode"] == eval_mode):
            return True
    return False

def add_processed_paper(paper_id, bot_num, tracking_data, eval_mode=False, s3_key=""):
    """Add a processed paper to the tracking data and update bot/paper mappings."""
    entry = {
        "paper_id": paper_id,
        "bot_num": bot_num,
        "processed_date": datetime.now().isoformat(),
        "eval_mode": eval_mode,
        "s3_result_key": s3_key,
        "model_used": model_name
    }
    tracking_data["processed_papers"].append(entry)
    
    # Update papers_by_bot mapping
    bot_key = str(bot_num)
    if bot_key not in tracking_data["papers_by_bot"]:
        tracking_data["papers_by_bot"][bot_key] = []
    
    paper_entry = {
        "paper_id": paper_id,
        "processed_date": datetime.now().isoformat(),
        "eval_mode": eval_mode,
        "s3_result_key": s3_key
    }
    tracking_data["papers_by_bot"][bot_key].append(paper_entry)
    
    # Update bots_by_paper mapping
    if paper_id not in tracking_data["bots_by_paper"]:
        tracking_data["bots_by_paper"][paper_id] = []
    
    bot_entry = {
        "bot_num": bot_num,
        "processed_date": datetime.now().isoformat(),
        "eval_mode": eval_mode,
        "s3_result_key": s3_key
    }
    tracking_data["bots_by_paper"][paper_id].append(bot_entry)

def get_unprocessed_papers(all_paper_ids, bot_num, tracking_data, eval_mode=False):
    """Get list of papers that haven't been processed by a specific bot."""
    unprocessed = []
    for paper_id in all_paper_ids:
        if not is_paper_already_processed(paper_id, bot_num, tracking_data, eval_mode):
            unprocessed.append(paper_id)
    return unprocessed

def get_bot_statistics(tracking_data):
    """Get statistics about which bots processed how many papers."""
    stats = {}
    for bot_num in range(NUM_BOTS):
        bot_key = str(bot_num)
        bot_papers = tracking_data["papers_by_bot"].get(bot_key, [])
        stats[bot_num] = {
            "total_papers": len(bot_papers),
            "eval_papers": len([p for p in bot_papers if p.get("eval_mode", False)]),
            "non_eval_papers": len([p for p in bot_papers if not p.get("eval_mode", False)])
        }
    return stats

def process_paper_for_bot(paper_id: str, bot_num: int, num_summaries: int, eval=True, threads=False, prefix='papers/', disable_s3=False, upload_prefix=''):
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
        obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=f'{prefix}{paper_id}.txt')
        paper_text = obj['Body'].read().decode('utf-8')
    except ClientError as e:
        print(f"Error getting object papers/{paper_id}.txt: {e}")
        return None
    
    print(f"generating summary with {len(paper_text)} characters")
    
    max_new_tokens_summary, max_new_tokens_tweet = 250, 26
    if threads:
        max_new_tokens_summary, max_new_tokens_tweet = 300, 60
        
    best_summary = ""
    if eval:
        all_summaries, best_summary_idx, evaluation = sum_eval(paper_text, tokenizer, model, model_name, num_summaries, max_new_tokens_summary)
        if best_summary_idx < 0 or best_summary_idx >= len(all_summaries):
            best_summary = random.choice(all_summaries)
        else:
            best_summary = all_summaries[best_summary_idx]
        evaluation = evaluation.replace('\nAnswer', '').replace('```', '')
    
    else:
        if 'qwen' in model_name.lower():
            best_summary = generate_summary(paper_text, tokenizer, model, bot_num)

        best_summary_idx = -1
        all_summaries = []
        evaluation = "Evaluation turned off"

    print(f"APP.PY: CALLING GENERATE TWEET")

    if threads:
        tweets = summary_splitting(best_summary, tokenizer, model, model_name, max_new_tokens=max_new_tokens_tweet, max_length=7000, bot_num=bot_num)
        real_tweets = []
        for t in tweets:
            print(f"t: {t}")
            real_tweet = t.split('\n')[0]
            print(f"real_tweet: {real_tweet}")
            real_tweets.append(t)

        result = {
            'status': 'success',
            'paper_id': paper_id,
            'bot_num': bot_num,
            'processed_date': datetime.now().isoformat(),
            'all_summaries': all_summaries,
            'best_summary_idx': best_summary_idx,
            'summary': best_summary,
            'evaluation': evaluation,
            'tweet': tweet,
            'chunks': chunks,
            'real_tweet_list': real_tweets,
            'model_used': model_name,
            'eval_mode': eval
        }
        print(f"Result: {result}")

    else:
        if 'mistral' in model_name.lower():
            tweet = generate_tweet_mistral(best_summary, tokenizer, model, max_new_tokens=max_new_tokens_tweet, bot_num=bot_num)
        elif 'olmo' in model_name.lower():
            tweet = generate_tweet_olmo(best_summary, tokenizer, model, max_new_tokens=max_new_tokens_tweet, bot_num=bot_num)
        elif 'qwen' in model_name.lower():
            tweet = generate_tweet(best_summary, tokenizer, model, max_new_tokens=max_new_tokens_tweet, bot_num=bot_num)
        elif 'llama' in model_name.lower():
            tweet = generate_tweet_llama(best_summary, tokenizer, model, max_new_tokens=max_new_tokens_tweet, bot_num=bot_num)

        real_tweet = tweet.split('\n')[0]
        real_tweet += f"\n Link: https://arxiv.org/abs/{paper_id}"

        print(f"evaluation is: {evaluation}")

        result = {
            'status': 'success',
            'paper_id': paper_id,
            'bot_num': bot_num,
            'processed_date': datetime.now().isoformat(),
            'all_summaries': all_summaries,
            'best_summary_idx': best_summary_idx,
            'summary': best_summary,
            'evaluation': evaluation,
            'tweet': tweet,
            'real_tweet': real_tweet,
            'model_used': model_name,
            'eval_mode': eval
        }

        print(f"Result: {result}")
    
    if not disable_s3:
        today = datetime.now().strftime("%Y-%m-%d")
        s3_key = f"{upload_prefix}/{model_name}/bot{bot_num}/{today}/{paper_id}.json"

        # if threads and eval:
        #     s3_key = s3_key.replace("results", "results-eval-threads")
        # elif threads and not eval:
        #     s3_key = s3_key.replace("results", "results-threads")

        print(f"s3_key = {s3_key}")
        
        try:
            s3.put_object(
                Bucket=os.environ.get('BUCKET_NAME'),
                Key=s3_key,
                Body=json.dumps(result, indent=2),
                ContentType='application/json'
            )
            print(f"Put result into s3 bucket :)")
            return s3_key

        except ClientError as e:
            print(f"Error putting object {s3_key} into s3 dir: {e}")
            return None

def get_all_papers_from_bucket(prefix):
    """
    Returns all paper keys. This excludes the paper path and extension,
    so it only keeps the arxiv paper ID.
    """
    response = s3.list_objects_v2(
        Bucket=os.environ.get('BUCKET_NAME'),
        Prefix=prefix
    )

    paper_keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key'].split("/")[-1].split('.txt')[0]
            if key:  # Skip empty keys
                paper_keys.append(key)
    
    return paper_keys

def main():
    """
    This randomly selects NUM_INDICES for each bot. So the bots end up posting
    about different papers. Now with comprehensive tracking of bot-paper relationships.
    """
    parser = argparse.ArgumentParser(description='Pick specific subset of papers or randomly pick them, generate summaries, evals, tweets.')
    parser.add_argument('--num_summaries', default=2, help='Number of summaries to generate for eval', type=int)
    parser.add_argument('--model_name', default="Qwen/Qwen3-4B")
    parser.add_argument('--no_eval', action='store_true', help='Turns off eval when this flag is added')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of already processed papers')
    parser.add_argument('--show_stats', action='store_true', help='Show bot processing statistics')
    parser.add_argument('--prefix', required=True, help='The prefix that indicates where this script should get all objects. Include / at end, like papers/')
    parser.add_argument('--threads', action='store_true', help='Put results into some threads folder')
    parser.add_argument('--disable_s3', action='store_true', help='Disables s3 uploads. Useful for testing.')
    parser.add_argument('--upload_prefix', required=True, help='The base folder path that you want to upload to. Example: results-eval-threads. Don\'t include the / at the end.')
    args = parser.parse_args()
    
    global model_name
    model_name = args.model_name
    
    # Load tracking data first
    tracking_data = load_processed_papers_tracker()
    
    # Show statistics if requested
    if args.show_stats:
        stats = get_bot_statistics(tracking_data)
        print("\n=== Bot Processing Statistics ===")
        for bot_num, stat in stats.items():
            print(f"Bot {bot_num}: {stat['total_papers']} total papers "
                  f"({stat['eval_papers']} with eval, {stat['non_eval_papers']} without eval)")
        print("=====================================\n")
    
    load_model()
    paper_ids = get_all_papers_from_bucket(args.prefix)
    print(f"Found {len(paper_ids)} papers in bucket")

    eval_mode = not args.no_eval
    # disable_s3 = not args.disable_s3
    threads = args.threads
    print(f"Eval mode: {eval_mode}")
    print(f"threads mode: {threads}")
    
    # Process papers for each bot
    processed_this_run = 0
    
    # TODO: CHANGE THIS BACK TO NUM_BOTS ONCE EVERYTHING WORKS. 
    for bot_num in range(NUM_BOTS):
        print(f"\n--- Processing for Bot {bot_num} ---")
        
        if args.force_reprocess:
            # Use all papers if forcing reprocess
            available_papers = paper_ids
        else:
            # Get unprocessed papers for this bot
            available_papers = get_unprocessed_papers(paper_ids, bot_num, tracking_data, eval_mode)
        
        print(f"Bot {bot_num}: {len(available_papers)} available papers")
        
        if len(available_papers) == 0:
            print(f"No unprocessed papers for bot {bot_num}")
            continue
            
        # Select papers for this bot
        num_to_select = len(available_papers)
        selected_papers = random.sample(available_papers, num_to_select)
        
        print(f"Bot {bot_num} selected papers: {selected_papers}")
        
        # Process each selected paper
        for paper_id in selected_papers:
            print(f"\nProcessing {paper_id} for bot {bot_num}")
            
            s3_key = process_paper_for_bot(paper_id, bot_num, args.num_summaries, eval_mode, threads, args.prefix, args.disable_s3, args.upload_prefix)
            
            if s3_key:
                # Add to in-memory tracking (batch save at end)
                add_processed_paper(paper_id, bot_num, tracking_data, eval_mode, s3_key)
                print(f"Added {paper_id} to tracking for bot {bot_num}")
                processed_this_run += 1
    
    # Save all tracking data at once at the end
    if processed_this_run > 0:
        save_processed_papers_tracker(tracking_data)
        print(f"\n=== Processing Complete ===")
        print(f"Processed {processed_this_run} new papers this run")
        print(f"Total papers tracked: {len(tracking_data['processed_papers'])}")
        
        # Show final statistics
        final_stats = get_bot_statistics(tracking_data)
        print("\n=== Final Bot Statistics ===")
        for bot_num, stat in final_stats.items():
            print(f"Bot {bot_num}: {stat['total_papers']} total papers "
                  f"({stat['eval_papers']} with eval, {stat['non_eval_papers']} without eval)")
        print("============================")
    else:
        print("\nNo new papers were processed this run.")

if __name__ == '__main__':
    main()