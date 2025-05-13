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

def load_paper(paper_path):
    """Load a paper from a PDF file"""
    text_content = ""
    try:
        with open(paper_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""
    return text_content

def generate_summary(text, tokenizer, model, max_length=200):
    """Generate a summary using Qwen3-3B"""
    # Truncate text if needed
    max_chars = 12000
    if len(text) > max_chars:
        print(f"Truncating text from {len(text)} to {max_chars} characters")
        text = text[:max_chars]
    
    # Create a prompt
    prompt = f"""Please provide a concise summary of the following research paper in no more than 200 words:

{text}

Summary:"""

    # Tokenize with conservative limits
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = inputs.to(model.device)
    
    # Generate summary
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up the output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.split("Summary:")[-1].strip()
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Generate paper summary using Qwen3-3B')
    parser.add_argument('--paper_path', required=True, help='Path to the paper PDF file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.paper_path):
        print(f"Error: File {args.paper_path} not found")
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
    
    # Load paper
    print(f"Loading paper from {args.paper_path}")
    paper_text = load_paper(args.paper_path)
    
    if not paper_text:
        print("Failed to extract text from the PDF")
        return
    
    print(f"Extracted {len(paper_text)} characters from PDF")
    print_memory_usage("After PDF load")
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(paper_text, tokenizer, model)
    
    # Print results
    print("\nGenerated Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {args.output_path}")
    
    # Clean up
    del mode