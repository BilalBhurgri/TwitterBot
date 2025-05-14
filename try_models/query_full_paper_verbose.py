import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import PyPDF2
import gc
import psutil
import time
import traceback

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[{label}] RAM Usage: {ram_usage:.2f} MB")
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[{label}] GPU Memory: {gpu_allocated:.2f} MB allocated")

def load_paper(paper_path):
    """Load a paper from a PDF file with better error handling"""
    text_content = ""
    try:
        with open(paper_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            print(f"PDF has {total_pages} pages")
            
            for page_num in range(total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # Debug: Check if page text is empty
                    if not page_text or page_text.strip() == "":
                        print(f"Warning: Page {page_num} returned empty text")
                    else:
                        text_content += page_text + "\n"
                        
                    # Print progress for large PDFs
                    if page_num % 10 == 0 and page_num > 0:
                        print(f"Processed {page_num}/{total_pages} pages")
                        
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
                    traceback.print_exc()
                    continue
                    
        # Verify we got some text
        if not text_content or text_content.strip() == "":
            print("ERROR: No text was extracted from the PDF")
            return ""
            
        print(f"Successfully extracted {len(text_content)} characters of text")
        
        # Debug: Show sample of extracted text
        print("Sample of extracted text (first 200 chars):")
        print(text_content[:200])
        
        return text_content
        
    except Exception as e:
        print(f"Error loading PDF: {e}")
        traceback.print_exc()
        return ""

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
    prompt = f"""Write an extremely concise summary of this paper. Focus only on key contributions and results.:

    PAPER TEXT:
    {text}

    INSTRUCTIONS:
    Write a concise summary of the above paper in about 200 words. Focus on key findings and contributions.
    DO NOT repeat the paper text verbatim.
    DO NOT include phrases like "this paper" or "the authors".
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
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

        # Get input length in tokens
        input_length = inputs.input_ids.shape[1]

        # Extract only the newly generated tokens
        generated_tokens = outputs[0][input_length:]

        # Decode only these new tokens to get the summary
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Check if outputs is empty or invalid
        if not hasattr(outputs, 'sequences') or outputs.sequences.size(0) == 0:
            print("ERROR: Model returned empty outputs")
            return "Model generation failed to produce output."
            
        # Decode with detailed logging
        print("Decoding output...")
        
        # Check shape of outputs
        print(f"Output sequence shape: {outputs.sequences.shape}")
        
        # Decode the first sequence
        full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f"Full decoded text length: {len(full_text)} characters")
        
        # Extract summary portion
        if "Summary:" in full_text:
            summary = full_text.split("Summary:")[-1].strip()
            print(f"Extracted summary of {len(summary)} characters")
        else:
            print("WARNING: 'Summary:' marker not found in output")
            summary = full_text  # Fall back to full text
            
        # Check for empty summary
        if not summary or summary.strip() == "":
            print("ERROR: Empty summary after extraction")
            return "Failed to generate a valid summary."
            
        return summary
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

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