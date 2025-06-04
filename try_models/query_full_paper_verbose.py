import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import gc
import psutil
import time
import traceback
import re
import json
from pathlib import Path
import sys

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
EXAMPLES_PATH = os.path.join(project_root, 'try_models','example_outputs', 'examples.json')

# Must come after project_root is appended
import data_processing.parse_paper_remove_math as parse_paper_remove_math
import data_processing.parse_paper as parse_paper

examples = {}

# Load examples from JSON
with open(EXAMPLES_PATH, 'r') as f:
    examples = json.load(f)

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[{label}] RAM Usage: {ram_usage:.2f} MB")
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[{label}] GPU Memory: {gpu_allocated:.2f} MB allocated")

def load_paper(paper_path):
    """Read the contents of a text file."""
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error reading text file: {str(e)}")
        return None

def generate_summary(text, tokenizer, model, max_new_tokens=250):
    """
    Generate a summary with extensive debugging. This assumes we use Qwen, 
    because its output after calling generate() is a sequence instead of just tokens.
    This same code wouldn't work for a mistral model. 
    """
    if not text or text.strip() == "":
        print("ERROR: Cannot generate summary from empty text")
        return "No text was provided for summarization."
    
    # Create a prompt
    prompt = f"""You will be given a scientific paper. Please write a 150-word summary based on the instructions.

Paper text:
{text}

Instructions:
Include key findings of the paper in your summary.
Make sure the summary is factually consistent with the paper. Do not include non-factual information.
Output the summary text in a single line and nothing else. Do not output your thought process.

Your summary:
"""

    print(f"Prompt created with {len(prompt)} characters")
    max_context = model.config.max_position_embeddings
    print(f"Max context length: {max_context}")
    max_prompt_len = max_context - max_new_tokens
    print(f"Max prompt length: {max_prompt_len}")
    if max_prompt_len < 0:
        print("ERROR: max_new_tokens is too long?")
        return "max_new_tokens is too long?"
    
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
        
        # Generate summary with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        # Setting return_dict_in_generate=True to get more debug info
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

        # # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        lines = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")

        summary = max(lines.splitlines(), key=len)
        
        # Verify summary
        if not summary or summary.strip() == "":
            print("ERROR: Generated summary is empty")
            return "Generated summary is empty"
            
        # Print results
        print("\nGenerated Summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        return summary
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def generate_summary_mistral(text, tokenizer, model, max_length=7000):
    """Generate a summary using Mistral model - uses 8K context window"""
    if not text or text.strip() == "":
        print("ERROR: Cannot generate summary from empty text")
        return "No text was provided for summarization."
    
    # Truncate text if needed
    max_chars = max_length
    if len(text) > max_chars:
        print(f"Truncating text from {len(text)} to {max_chars} characters")
        text = text[:max_chars]
    
    # Create a prompt
    prompt = f"""<s>[INST] Write a concise 200-word summary of this research paper. Focus on key findings and contributions. Write it like a tweet - engaging and accessible to a general audience. Do not include phrases like "this paper" or "the authors". Only use English.

Paper text:
{text} [/INST]</s>"""

    print(f"Prompt created with {len(prompt)} characters")
    
    try:
        # Tokenize with conservative limits
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000)
        inputs = inputs.to(model.device)
        
        print(f"Input tokenized to {inputs.input_ids.shape[1]} tokens")
        
        # Generate summary with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            min_new_tokens=50,  # Force at least some generation
            do_sample=True,
            temperature=0.8,  # Slightly higher temperature
            top_p=0.95,
            top_k=50,  # Increased top_k
            repetition_penalty=1.2,  # Increased repetition penalty
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Get input length in tokens
        input_length = inputs.input_ids.shape[1]
        print(f"Input length: {input_length}")
        print(f"Output shape: {outputs.shape}")
        
        # Decode the full output first to debug
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full output: {full_output}")
        
        # Decode only the newly generated tokens
        summary = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        print(f"Generated summary: {summary}")
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return summary
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def generate_eval(text, summaries, tokenizer, model):
    if not summaries:
        print("ERROR: Cannot generate evaluation from empty summary")
        return "No summary was provided for evaluation."
    
    """Generate a evaluation with extensive debugging"""
    if not text or text.strip() == "":
        print("ERROR: Cannot generate evaluation from empty text")
        return "No text was provided for evaluation."
    
    # Create a prompt
    prompt = f"""You will be given several summaries written for the same research paper. Your task is to rate the summaries on two metrics.

Source Text:
{text}

"""

    for i in range(len(summaries)):
        prompt = prompt + f"""

Summary {i}:
{summaries[i]}

"""

    prompt = prompt + f"""

Criteria:

1. Factual Consistency (1-3): Does the summary only contain facts supported by the source text?
- 1 (Inconsistent): Major errors or many minor errors
- 2 (Overall consistent): At most one minor error
- 3 (Consistent): All facts supported

2. Engagingness (1-3): Is the summary interesting to most audiences?
- 1 (Dull): Only interesting to specialists
- 2 (Somewhat interesting): Engages those familiar with the field
- 3 (Interesting): Engages general audiences regardless of expertise

Instructions:

1. Design up to 3 evaluation steps based on the evaluation criteria. Output each step on a new line.
2. Based on your evaluation steps, output the factual consistency and engagingness scores of each summary on a new line.
3. Choose the best summary based on your evaluation. Output its index on a new line.
4. Output nothing else. Do not output your thought process.

Evaluation Steps:
"""

    print(f"Prompt created with {len(prompt)} characters")

    max_context = model.config.max_position_embeddings
    print(f"Max context length: {max_context}")
    max_new_tokens= 100 + 100 * len(summaries)
    max_prompt_len = max_context - max_new_tokens
    print(f"Max prompt length: {max_prompt_len}")
    if max_prompt_len < 0:
        print("ERROR: Too many summaries to evaluate at the same time")
        return "Too many summaries to evaluate at the same time"
    
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
        
        # Generate summary with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        # Setting return_dict_in_generate=True to get more debug info
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        # # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        eval = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return eval.strip()
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def generate_eval_mistral(text, summary, tokenizer, model, max_length=7000):
    return "not implemented"

# Takes in paper text, returns best summary & eval
def sum_eval(paper_text, tokenizer, model, model_name, num_summaries=2):
    summaries = [None] * num_summaries
    for i in range(num_summaries):
        # Generate summary
        print("Generating summary...")
        if 'mistral' in model_name.lower():
            summaries[i] = generate_summary_mistral(paper_text, tokenizer, model)
        else:
            # qwen code
            summaries[i] = generate_summary(paper_text, tokenizer, model)
        
        # Verify summary
        if not summaries[i] or summaries[i].strip() == "":
            print("ERROR: Generated summary is empty")
            return summaries[:i], -1, "ERROR: Generated summary is empty"

    # Generate evaluation
    print("Generating evaluation...")
    if 'mistral' in model_name.lower():
        eval = generate_eval_mistral(paper_text, summaries, tokenizer, model)
    else:
        # qwen code
        eval = generate_eval(paper_text, summaries, tokenizer, model)
    
    # Verify summary
    if not eval or eval.strip() == "":
        print("ERROR: Generated evaluation is empty")
        return summaries, -1, "ERROR: Generated evaluation is empty"
    
    # Print results
    print("\nGenerated Evaluation:")
    print("-" * 50)
    print(eval)
    print("-" * 50)

    # Extract index of best summary
    lines = eval.splitlines()
    numLines = [line for line in lines if re.search(r'\b(Best|Answer)\b', line) and re.search(r'\d', line)]
    if numLines:
        print("\nExtracted Lines:")
        for line in numLines:
            print(line)
        numbers = [re.sub(r'\D', '', line) for line in numLines]
        if not numbers:
            print("ERROR: Cannot find number in extracted lines")
            return summaries, -1, eval
        print("\nExtracted Numbers:")
        for number in numbers:
            print(number)
        for i in range(len(numbers) - 1, -1, -1):
            best_idx = int(numbers[i])
            if best_idx >= 0 and best_idx < len(summaries):
                print(f"Best idx: {best_idx}")
                return summaries, best_idx, eval
    
    numLines = [line for line in lines if re.fullmatch(r'\s*[0-9]\s*', line)]
    if not numLines:
        print("ERROR: Cannot find line with single number/best: number/answer: number in evaluation")
        return summaries, -1, eval
    print("\nExtracted Lines:")
    for line in numLines:
        print(line)
    numbers = [re.sub(r'\D', '', line) for line in numLines]
    if not numbers:
        print("ERROR: Cannot find number in extracted lines")
        return summaries, -1, eval
    print("\nExtracted Numbers:")
    for number in numbers:
        print(number)
    for i in range(len(numbers) - 1, -1, -1):
        best_idx = int(numbers[i])
        if best_idx >= 0 and best_idx < len(summaries):
            print(f"Best idx: {best_idx}")
            return summaries, best_idx, eval
    
    print("No index within bounds of summaries")
    return summaries, -1, eval

def main():
    parser = argparse.ArgumentParser(description='Generate paper summary using Qwen/Qwen3-1.7B')
    parser.add_argument('--paper_path', required=True, help='Path to the paper\'s text file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary')
    parser.add_argument('--eval_path', default=None, help='Path to save the evaluation')
    parser.add_argument('--model_name', default="Qwen/Qwen3-1.7B", help='Model to use (default: Qwen/Qwen3-1.7B)')
    parser.add_argument('--num_summaries', default=2, help='Number of summaries to generate', type=int)
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.paper_path):
        print(f"Error: File {args.paper_path} not found")
        return
    
    print(f"Starting script with parameters:")
    print(f"  - Paper: {args.paper_path}")
    print(f"  - Output: {args.output_path}")
    print(f"  - Model: {args.model_name}")
    
    print_memory_usage("Starting")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer with extra error handling
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

        # After loading the model
        print(f"Model device: {next(model.parameters()).device}")
        
        # During generation, time the operation
        start_time = time.time()
        
        # Verify model and tokenizer are valid
        if not hasattr(model, 'generate'):
            print("ERROR: Loaded model doesn't have generate method")
            return
            
        # Load paper
        print(f"Loading paper from {args.paper_path}")
        paper_text = load_paper(args.paper_path)
        
        if not paper_text:
            print("ERROR: Failed to extract text from the PDF")
            return
        
        print(f"Successfully extracted {len(paper_text)} characters from PDF")
        print_memory_usage("After PDF load")

        if args.eval_path:
            summaries, best_idx, eval = sum_eval(paper_text, tokenizer, model, args.model_name, args.num_summaries)
            if best_idx is None:
                best_summary = summaries[0]
            else:
                best_summary = summaries[best_idx]
        else:
            if 'mistral' in model_name.lower():
                best_summary = generate_summary_mistral(paper_text, tokenizer, model)
            else:
                best_summary = generate_summary(paper_text, tokenizer, model)
            
        if best_summary is None:
            print("ERROR: Best summary not returned")
            return
        
        if args.output_path:
            try:
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    f.write(best_summary)
                print(f"Summary saved to {args.output_path}")
            except Exception as e:
                print(f"ERROR saving output: {e}")
        
        if args.eval_path:
            if eval is None:
                print("ERROR: Evaluation not returned")
                return
            try:
                with open(args.eval_path, 'w', encoding='utf-8') as f:
                    f.write(eval)
                print(f"Evaluation saved to {args.eval_path}")
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
        generation_time = time.time() - start_time
        print(f"Generation took {generation_time:.2f} seconds")
        
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
