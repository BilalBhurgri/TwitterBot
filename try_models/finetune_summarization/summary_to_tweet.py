import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import gc
import psutil
import time
import traceback
import json
from pathlib import Path
import sys

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[{label}] RAM Usage: {ram_usage:.2f} MB")
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[{label}] GPU Memory: {gpu_allocated:.2f} MB allocated")

def generate_tweet_mistral(summary: str, tokenizer, model, max_length=7000):
    """
    Generate a tweet from a summary using Mistral model
    """
    if not summary or summary.strip() == "":
        print("ERROR: Cannot generate tweet from empty summary")
        return "No summary was provided for tweet generation."
    
    # Create a prompt
    prompt = f"""<s>[INST] Convert this research paper summary into an engaging tweet. Make it accessible to a general audience, highlight the most interesting finding, and keep it under 280 characters. Only use English.

        Summary:
        {summary} [/INST]</s>"""

    print(f"Prompt created with {len(prompt)} characters")
    
    try:
        # Tokenize with conservative limits
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000)
        inputs = inputs.to(model.device)
        
        print(f"Input tokenized to {inputs.input_ids.shape[1]} tokens")
        
        # Generate tweet with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=280,  # Twitter's character limit
            min_new_tokens=50,  # Force at least some generation
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Get input length in tokens
        input_length = inputs.input_ids.shape[1]
        print(f"Input length: {input_length}")
        print(f"Output shape: {outputs.shape}")
        
        # Decode only the newly generated tokens
        tweet = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        print(f"Generated tweet: {tweet}")
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return tweet.strip()
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating tweet: {str(e)}"

def generate_tweet_qwen(summary: str, tokenizer, model, max_new_tokens=280):
    """
    Generate a tweet from a summary using Qwen model
    """
    if not summary or summary.strip() == "":
        print("ERROR: Cannot generate tweet from empty summary")
        return "No summary was provided for tweet generation."
    
    # Create a prompt
    prompt = f"""
    Convert this research paper summary into an engaging tweet. Follow these instructions:

    Instructions:
    - Make it accessible to a general audience
    - Highlight the most interesting finding
    - Keep it under 280 characters
    - Do not include phrases like "this paper" or "the authors"
    - Only use English
    - Output the tweet text only and nothing else

    Summary:
    {summary}

    Your tweet:
    """

    print(f"Prompt created with {len(prompt)} characters")
    max_context = model.config.max_position_embeddings
    max_prompt_len = max_context - max_new_tokens
    
    try:
        print("Tokenizing input...")
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
            padding=False
        )
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        print(f"Input tokenized to {input_ids.shape[1]} tokens")
        
        # Generate tweet with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        tweet = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return tweet.strip()
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating tweet: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate tweet from paper summary')
    parser.add_argument('--summary_path', required=True, help='Path to the summary text file')
    parser.add_argument('--output_path', default=None, help='Path to save the generated tweet')
    parser.add_argument('--model_name', default="Qwen/Qwen3-4B", help='Model to use (default: Qwen/Qwen3-4B)')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.summary_path):
        print(f"Error: File {args.summary_path} not found")
        return
    
    print(f"Starting script with parameters:")
    print(f"  - Summary: {args.summary_path}")
    print(f"  - Output: {args.output_path}")
    print(f"  - Model: {args.model_name}")
    
    print_memory_usage("Starting")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    model_name = args.model_name
    print(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        # Load in half-precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully")
        print_memory_usage("After model load")

        # Load summary
        print(f"Loading summary from {args.summary_path}")
        with open(args.summary_path, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        if not summary:
            print("ERROR: Failed to read summary")
            return
        
        print(f"Successfully loaded summary of {len(summary)} characters")
        print_memory_usage("After summary load")

        # Generate tweet
        print("Generating tweet...")
        if 'mistral' in model_name.lower():
            tweet = generate_tweet_mistral(summary, tokenizer, model)
        else:
            tweet = generate_tweet_qwen(summary, tokenizer, model)
        
        # Verify tweet
        if not tweet or tweet.strip() == "":
            print("ERROR: Generated tweet is empty")
            return
            
        # Print results
        print("\nGenerated Tweet:")
        print("-" * 50)
        print(tweet)
        print("-" * 50)
        
        if args.output_path:
            try:
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    f.write(tweet)
                print(f"Tweet saved to {args.output_path}")
            except Exception as e:
                print(f"ERROR saving output: {e}")
        
        # Clean up
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print_memory_usage("After cleanup")
        print("Script completed successfully")
        
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()