from flask import Flask, jsonify, request
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

# Load environment variables
load_dotenv()

app = Flask(__name__)

s3 = boto3.client('s3', region_name='us-west-1')
response = s3.list_buckets()

# papers_response = s3.list_objects_v2(
#     Bucket=os.environ.get('BUCKET_NAME'),
#     Prefix='papers/'
# )

# papers_dict = {}

# for obj in papers_response.get('Contents', []):
#     print(f"File: {obj['Key']}")
#     print(f"Size: {obj['Size']} bytes")
#     print(f"Modified: {obj['LastModified']}")
#     print("---")
#     paper_id = obj['Key'].split('/')[-1].split('.')[0]
#     print(f"Paper ID: {paper_id}")
#     papers_text = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=obj['Key'])
#     papers_dict[obj['Key']] = paper_id
#     papers_dict[obj['Key']] = papers_text['Body'].read().decode('utf-8')

# # Temp comment: confirm that paper text prints properly 
# print(papers_dict['papers/2412.00857.txt'] )
    

# output bucket names 
# print('Existing buckets:')
# for bucket in response['Buckets']:
#     print(f'  {bucket["Name"]}')

# Global variables for model and tokenizer
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running',
        'model_loaded': model is not None
    })

@app.route('/process-paper/<key>', methods=['GET'])
def process_paper(key):
    """Process a paper and generate a summary"""
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

@app.route('/set-model', methods=['POST'])
def set_model():
    """Set the model to use"""
    global model, tokenizer, model_name
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No model name provided'
            }), 400

        new_model_name = data['model_name']
        # Reset model and tokenizer
        model = None
        tokenizer = None
        model_name = new_model_name
        
        # Load new model
        load_model()
        
        return jsonify({
            'status': 'success',
            'message': f'Model set to {model_name}'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.getenv('PORT', 5000))
    # Run the app
    app.run(host='0.0.0.0', port=port) 