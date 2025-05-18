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
import parse_paper_remove_math
import parse_paper

examples = {}
# Load examples from JSON
with open('./example_outputs/examples.json', 'r') as f:
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
    text = parse_paper.extract_text_from_xml(paper_path)
    return text

def generate_summary(text, tokenizer, model, max_length=200):
    """Generate a summary with extensive debugging"""
    if not text or text.strip() == "":
        print("ERROR: Cannot generate summary from empty text")
        return "No text was provided for summarization."
    
    # Truncate text if needed
    max_chars = 20000
    if len(text) > max_chars:
        print(f"Truncating text from {len(text)} to {max_chars} characters")
        text = text[:max_chars]
    
    # Create a prompt
    prompt = f"""
    EXAMPLE:
    {examples["good_formal_example"]}

    INSTRUCTIONS:
    Write a 200 word summary of this paper like a twitter post. Focus on key findings and contributions.
    DO NOT repeat the paper text verbatim.
    DO NOT include phrases like "this paper" or "the authors".
    ONLY USE ENGLISH!
    DO NOT reuse the example output format.

    PAPER TEXT:
    {text}
    """

    print(f"Prompt created with {len(prompt)} characters")
    
    try:
        # Tokenize with conservative limits
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20000)
        inputs = inputs.to(model.device)
        
        print(f"Input tokenized to {inputs.input_ids.shape[1]} tokens")
        
        # Generate summary with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        # Setting return_dict_in_generate=True to get more debug info
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

        # Get the sequences
        sequences = outputs.sequences
        # # print(f"Shape of sequences: {sequences.shape}")

        # # Get input length in tokens
        input_length = inputs.input_ids.shape[1]
        # print(f"Input length: {input_length}")
        # # Extract only the newly generated tokens for the first sequence
        generated_tokens = sequences[0, input_length:]

        # # Decode only the newly generated tokens
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # generation_time = time.time() - start_time
        # print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Step 1: Get your original input text as a string
        # original_input_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        # print(f"Original input text: {original_input_text}")

        # Step 2: Get the full generated text as a string
        # sequences = outputs.sequences
        # full_generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        # print(f"Full generated text: {full_generated_text}")

        # Step 3: Remove the input text from the beginning of the generated text
        # if full_generated_text.startswith(original_input_text):
        #     generated_only = full_generated_text[len(original_input_text):]
        #     print(f"Generated text only (string method): {generated_only}")
        # else:
        #     print("WARNING: Generated text doesn't start with the input text exactly")
        #     # Try a more fuzzy approach
        #     print("Attempting fuzzy matching...")
            
        #     # Print a portion of both for comparison
        #     print(f"Input starts with: {original_input_text[:100]}...")
        #     print(f"Output starts with: {full_generated_text[:100]}...")
        
            
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20000)
        inputs = inputs.to(model.device)
        
        print(f"Input tokenized to {inputs.input_ids.shape[1]} tokens")
        
        # Generate summary with verbose logging
        print("Starting generation...")
        start_time = time.time()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Get the sequences
        # sequences = outputs.sequences
        
        # # Get input length in tokens
        input_length = inputs.input_ids.shape[1]
        
        # # Extract only the newly generated tokens for the first sequence
        # generated_tokens = sequences[0, input_length:]

        # Decode only the newly generated tokens
        summary = tokenizer.decode(outputs[0, input_length: ].tolist(), skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return summary
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def test_load_paper():
    parser = argparse.ArgumentParser(description='Generate paper summary using Qwen/Qwen3-1.7B')
    parser.add_argument('--paper_path', required=True, help='Path to the paper PDF file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary')
    parser.add_argument('--model_name', default="Qwen/Qwen3-1.7B", help='Model to use (default: Qwen/Qwen3-1.7B)')
    args = parser.parse_args()

    paper_text = load_paper(args.paper_path)


def main():
    parser = argparse.ArgumentParser(description='Generate paper summary using Qwen/Qwen3-1.7B')
    parser.add_argument('--paper_path', required=True, help='Path to the paper PDF file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary')
    parser.add_argument('--model_name', default="Qwen/Qwen3-1.7B", help='Model to use (default: Qwen/Qwen3-1.7B)')
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
        
        # Generate summary
        print("Generating summary...")
        if 'mistral' in model_name.lower():
            summary = generate_summary_mistral(paper_text, tokenizer, model)
        else:
            # qwen code
            summary = generate_summary(paper_text, tokenizer, model)
        
        # Verify summary
        if not summary or summary.strip() == "":
            print("ERROR: Generated summary is empty")
            return
            
        # Print results
        print("\nGenerated Summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        if args.output_path:
            try:
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to {args.output_path}")
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