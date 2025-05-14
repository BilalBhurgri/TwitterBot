"""UNTESTED: Eval implementation based on query_full_paper_smaller.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import PyPDF2
import gc
import psutil

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[{label}] RAM Usage: {ram_usage:.2f} MB")
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[{label}] GPU Memory: {gpu_allocated:.2f} MB allocated")

def evaluate_summary(document_text, summmary_text, tokenizer, model):
    """Generate a summary using Qwen3-3B"""
    # Truncate text if needed
    max_chars = 12000
    if len(document_text) > max_chars:
        print(f"Truncating text from {len(document_text)} to {max_chars} characters")
        document_text = document_text[:max_chars]
    
    # Create a prompt
    prompt = f"""You will be given one summary tweet written for a research paper.

Your task is to rate the tweet on two metrics. Read these instructions carefully and refer back as needed.

Evaluation Criteria:

1. Factual Consistency (1-3): Does the tweet only contain facts supported by the source text?
- 1 (Inconsistent): Major errors or many minor errors
- 2 (Overall consistent): At most one minor error
- 3 (Consistent): All facts supported

2. Engagingness (1-3): Is the tweet interesting to most audiences?
- 1 (Dull): Only interesting to specialists
- 2 (Somewhat interesting): Engages those familiar with the field
- 3 (Interesting): Engages general audiences regardless of expertise

Evaluation Steps:

1. Read the source text and identify its key points.
2. Read the tweet. Check for factual consistency and engagingness.
3. Return two scores as: (Factual Consistency, Engagingness)

Example:

Source Text:
{document_text}

Summary:
{summary_text}

Evaluation Form:
(Factual Consistency, Engagingness):
"""

    # Tokenize with conservative limits
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = inputs.to(model.device)
    
    # Generate summary
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up the output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    eval = result.split("Evaluation:")[-1].strip()
    
    return eval

def main():
    parser = argparse.ArgumentParser(description='Evaluate summary using Qwen3-3B')
    parser.add_argument("--document_path", required=True, help="Path to full document text file")
    parser.add_argument("--summary_path", required=True, help="Path to summary text file")
    parser.add_argument('--output_path', default=None, help='Path to save the evaluation')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.document_path):
        print(f"Error: File {args.document_path} not found")
        return

    # Check if file exists
    if not os.path.exists(args.summary_path):
        print(f"Error: File {args.summary_path} not found")
        return
    
    print_memory_usage("Starting")
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen3-3B"  # Using the 3B model instead of 8B
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load in half-precision to save memory (no need for 4-bit quantization with 3B model)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("Model loaded successfully")
    print_memory_usage("After model load")
    
    # Read files
    with open(args.document_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    if not document_text:
        print("Failed to read document text")
        return
    
    with open(args.summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
    
    if not summary_text:
        print("Failed to read summary text")
        return
    
    # Generate summary
    print("Evaluating...")
    eval = evaluate_summary(document_text, summmary_text, tokenizer, model)
    
    # Print results
    print("\nGenerated Evaluation:")
    print("-" * 50)
    print(eval)
    print("-" * 50)
    
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write(eval)
        print(f"Evaluation saved to {args.output_path}")
    
    # Clean up
    del mode