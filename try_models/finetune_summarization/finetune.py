import os
import torch
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
import json

def load_twitter_data(file_path="twitter_training_data.json"):
    """Load scraped Twitter data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training examples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found. Using sample dataset instead.")
        # Fallback to tweet_eval dataset
        dataset = load_dataset("tweet_eval", "sentiment")
        return dataset['train'][:100]  # Take first 100 examples

def load_huggingface_dataset(dataset_name, split="train", subset=None, num_samples=None):
    """
    Load dataset from Hugging Face Hub
    
    Args:
        dataset_name: Name of the dataset (e.g., "cnn_dailymail", "xsum")
        split: Dataset split to use ("train", "validation", "test")
        subset: Dataset subset/config (e.g., "3.0.0" for cnn_dailymail)
        num_samples: Limit number of samples (None for all)
    
    Returns:
        List of examples from the dataset
    """
    print(f"Loading {dataset_name} from Hugging Face...")
    
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"Loaded {len(dataset)} examples from {dataset_name}")
        return list(dataset)
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        print("Falling back to tweet_eval dataset")
        fallback = load_dataset("tweet_eval", "sentiment", split="train")
        return list(fallback.select(range(100)))

def format_training_data(data, target_personality="elonmusk", data_source="scraped"):
    """
    Convert different data sources into instruction format
    
    Args:
        data: List of examples (format depends on data_source)
        target_personality: The personality style to emulate
        data_source: "scraped", "summarization", "twitter", or "custom"
    """
    formatted_examples = []
    
    for example in data:
        input_text = ""
        target_text = ""
        
        if data_source == "scraped":
            # From your scraped Twitter data
            original_text = example.get('text', '')
            summary = example.get('summary', '')
            
            if original_text and summary:
                input_text = f"Summarize this content in the style of @{target_personality}: {original_text}"
                target_text = summary[:280]  # Twitter length limit
                
        elif data_source == "summarization":
            # From datasets like cnn_dailymail, xsum
            article = example.get('article', '') or example.get('document', '')
            summary = example.get('highlights', '') or example.get('summary', '')
            
            if article and summary:
                input_text = f"Rewrite this summary in Twitter style like @{target_personality}: {summary}"
                target_text = summary[:280]
                
        elif data_source == "twitter":
            # From tweet_eval or similar Twitter datasets
            text_content = str(example.get('text', ''))
            if text_content:
                input_text = f"Rewrite this in the style of @{target_personality}: {text_content}"
                target_text = text_content[:100] + "..." if len(text_content) > 100 else text_content
                
        elif data_source == "custom":
            # Custom format with 'input' and 'output' keys
            input_text = example.get('input', '') or example.get('instruction', '')
            target_text = example.get('output', '') or example.get('response', '')
        
        if input_text and target_text:
            formatted_examples.append({
                "input_text": input_text,
                "target_text": target_text
            })
    
    return formatted_examples

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tokenize inputs and targets"""
    
    # Tokenize inputs
    inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    inputs["labels"] = targets["input_ids"]
    return inputs

def main():
    # Model setup
    model_name = "google/flan-t5-base"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v", "k", "o", "wi", "wo"]  # T5 specific modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully!")
    model.print_trainable_parameters()
    
    # Choose your data source:
    
    # Option 1: Load from scraped Twitter data
    # raw_data = load_twitter_data("twitter_training_data.json")
    # formatted_data = format_training_data(raw_data, target_personality="elonmusk", data_source="scraped")
    
    # Option 2: Load from Hugging Face summarization dataset
    # raw_data = load_huggingface_dataset("cardiffnlp/tweet_eval", "3.0.0", split="train", num_samples=1000)
    # formatted_data = format_training_data(raw_data, target_personality="elonmusk", data_source="summarization")
    
    # Option 3: Load from Twitter dataset
    raw_data = load_huggingface_dataset("tweet_eval", subset="sentiment", num_samples=500)
    formatted_data = format_training_data(raw_data, target_personality="elonmusk", data_source="twitter")
    
    # Option 4: Load custom instruction dataset
    # raw_data = load_huggingface_dataset("databricks/databricks-dolly-15k", num_samples=1000)
    # formatted_data = format_training_data(raw_data, target_personality="elonmusk", data_source="custom")
    
    print(f"Formatted {len(formatted_data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./twitter-personality-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb logging
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model("./twitter-personality-final")
    tokenizer.save_pretrained("./twitter-personality-final")
    
    print("Training completed! Model saved to ./twitter-personality-final")

def test_model(model_path="./twitter-personality-final"):
    """Test the trained model with sample inputs"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    test_inputs = [
        "Summarize this content in the style of @elonmusk: The new AI research paper shows significant improvements in language understanding capabilities.",
        "Rewrite this in Twitter style like @elonmusk: We are excited to announce our latest breakthrough in artificial intelligence."
    ]
    
    for input_text in test_inputs:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {input_text}")
        print(f"Output: {generated_text}")

if __name__ == "__main__":
    main()
    
    # Uncomment to test the trained model
    # test_model()