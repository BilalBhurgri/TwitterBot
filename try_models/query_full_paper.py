import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import PyPDF2

def load_paper(paper_path):
    """Load a paper from a PDF file"""
    text_content = ""
    try:
        with open(paper_path, 'rb') as pdf_file:  # Remove encoding parameter
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
        return text_content
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

def generate_summary(text, tokenizer, model, max_length=200):
    """Generate a summary using Qwen3-8B"""
    # Create a prompt that encourages concise summarization
    prompt = f"""Please provide a concise summary of the following research paper in no more than 200 words. Focus on the key findings, methodology, and significance:

        {text}

        Summary:"""

    # Reduce token count to save memory
    NUM_TOKENS = 8000  # Reduced from 15000
    
    # Tokenize the input with shorter context
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=NUM_TOKENS)
    inputs = inputs.to(model.device)

    # Generate summary with reduced parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,  # Reduced from NUM_TOKENS
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
    parser.add_argument('--paper_path', required=True, help='Path to the paper PDF file')
    parser.add_argument('--output_path', default=None, help='Path to save the summary (optional)')
    parser.add_argument('--use_8bit', action='store_true', help='Use 8-bit quantization to reduce memory usage')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.paper_path):
        print(f"Error: File {args.paper_path} not found")
        return

    # Load model and tokenizer with memory optimization
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use 8-bit quantization if requested to reduce memory usage
    if args.use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True  # Add low memory usage flag
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Try running with --use_8bit to reduce memory usage")
            return

    # Load paper
    paper_text = load_paper(args.paper_path)
    
    if not paper_text:
        print("Failed to extract text from the PDF. Check the file format.")
        return
    
    # Generate summary
    try:
        summary = generate_summary(paper_text, tokenizer, model)
        
        # Print or save results
        print("\nGenerated Summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU ran out of memory. Try using --use_8bit option or a smaller model.")
        else:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()