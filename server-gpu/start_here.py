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
from datetime import datetime

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import the paper processing functions
from try_models.query_full_paper_verbose import (
    load_paper,
    generate_summary,
    generate_summary_mistral,
    generate_eval,
    generate_eval_mistral
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

def process_paper_for_bot(paper_key, eval=False):
    """
    Process a paper and generate a summary, tweet, and optionally an eval
    Params: 
    key: paper_key is same as arxiv ID
    eval: bool. If True, eval happens.
    """
    try:
        # Check if model is loaded
        if model is None:
            load_model()

        paper_id = key
        obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=f'papers/{paper_id}.txt')
        paper_text = obj['Body'].read().decode('utf-8')
        
        print(f"generating summary with {paper_text}")
        
        # Generate summary
        print(f"APP.PY: CALLING GENERATE SUMMARY")
        summary = generate_summary(paper_text, tokenizer, model)
        if not summary:
            print(f"APP.PY: WARNING, SUMMARY IS EMPTY SO TWEET WILL BE EMPTY TOO")
        
        # Generate evaluation if requested
        # TODO: decide on what to do with this 
        evaluation = None
        # if the parameter doesnt exist, it returns default value 'false'.
        evaluate = request.args.get('evaluate', 'false').lower() == 'true'
        if evaluate:
            evaluation = generate_eval(paper_text, summary, tokenizer, model)

        print(f"APP.PY: CALLING GENERATE TWEET")
        tweet = generate_tweet_qwen(summary, tokenizer, model, max_new_tokens=300)
        real_tweet = tweet.split('Answer:')[-1]

        result = jsonify({
            'status': 'success',
            'summary': summary,
            'evaluation': evaluation,
            'tweet': tweet,
            'real_tweet': real_tweet,
        })

        today = datetime.now().strftime("%Y-%m-%d")
        s3_key = f"results/{model_name}/bot{bot_num}"
        s3.put_object(
            Bucket=os.environ.get('BUCKET_NAME'),
            Key=s3_key,
            Body=json.dumps(result_data, indent=2),
            ContentType='application/json'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



