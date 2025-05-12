import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import PyPDF2

NUM_TOKENS = 15000

def load_paper(paper_path):
    """Load a paper from a text file"""
    text_content = ""
    with open(paper_path, 'rb', encoding='utf-8') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()
    return text_content

def generate_summary(text, tokenizer, model, max_length=200):
    """Generate a summary using Qwen3-8B"""
    # Create a prompt that encourages concise summarization
    prompt = f"""Please provide a concise summary of the following research paper in no more than 200 words. Focus on the key findings, methodology, and significance:

        {text}

        Summary:"""

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, model_max_length=NUM_TOKENS)
    inputs = inputs.to(model.device)

    # Generate summary
    outputs = model.generate(
        **inputs,
        max_new_tokens=NUM_TOKENS,
        min_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean up the output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the summary part after "Summary:"
    summary = summary.split("Summary:")[-1].strip()
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Generate paper summary using Qwen3-8B')
    parser.add_argument('--paper_path', required=True, help='Path to the paper text file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary (optional)')
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load paper
    paper_text = load_paper(args.paper_path)
    
    # Generate summary
    summary = generate_summary(paper_text, tokenizer, model)
    
    # Print or save results
    print("\nGenerated Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

if __name__ == "__main__":
    main()
