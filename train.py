#!/usr/bin/env python3

import os
import re
import argparse
import torch
import glob
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on your writing samples")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Directory containing .txt files with your writing samples"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/final_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        help="Base model to fine-tune (e.g., gpt2, gpt2-medium, gpt2-large)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.15,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true",
        help="Use mixed precision training (requires GPU with CUDA)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to cache models and datasets"
    )
    
    return parser.parse_args()

def setup_directories(args):
    """Set up necessary directories"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

def load_and_clean_data(data_dir):
    """Load and clean text data from the data directory"""
    samples = []
    
    # Get all .txt files recursively
    all_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    
    if not all_files:
        raise ValueError(f"No .txt files found in {data_dir}. Please add your writing samples as .txt files.")
    
    for filepath in all_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Basic cleaning
                content = re.sub(r'https?://\S+|www\.\S+', '', content)  # Remove URLs
                content = re.sub(r'<.*?>', '', content)  # Remove HTML tags
                content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
                
                if content:  # Only add non-empty content
                    samples.append(content)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
    
    return samples

def segment_text(text, tokenizer, max_length=512):
    """Segment text into chunks for training"""
    # Simple paragraph-based segmentation
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Combine short paragraphs and split long ones
    segments = []
    current_segment = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if adding this paragraph would exceed max_length
        combined = current_segment + "\n\n" + para if current_segment else para
        combined_tokens = len(tokenizer.encode(combined))
        
        if combined_tokens > max_length and current_segment:
            segments.append(current_segment)
            current_segment = para
        else:
            current_segment = combined
    
    # Add the last segment if it exists
    if current_segment:
        segments.append(current_segment)
        
    return segments

def prepare_datasets(samples, tokenizer, max_length, val_split):
    """Prepare training and validation datasets"""
    # Segment all samples
    all_segments = []
    for sample in samples:
        segments = segment_text(sample, tokenizer, max_length)
        all_segments.extend(segments)
    
    print(f"Created {len(all_segments)} text segments for training")
    
    # Create train/validation splits
    train_segments, val_segments = train_test_split(all_segments, test_size=val_split, random_state=42)
    
    print(f"Training segments: {len(train_segments)}")
    print(f"Validation segments: {len(val_segments)}")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_segments})
    val_dataset = Dataset.from_dict({"text": val_segments})
    
    # Combine into a DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Function to tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def train_model(args, tokenized_dataset, tokenizer):
    """Train the model on the prepared dataset"""
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're using standard language modeling (not masked)
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch sizes
        learning_rate=args.learning_rate,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,  # Keep only the 3 most recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=args.fp16 and torch.cuda.is_available(),  # Use mixed precision if GPU is available and fp16 is enabled
        report_to="tensorboard",
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up necessary directories
    setup_directories(args)
    
    # Check if data directory exists and contains files
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' doesn't exist")
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu" and args.fp16:
        print("Warning: fp16 is enabled but a GPU is not available. Disabling fp16.")
        args.fp16 = False
    
    print(f"Loading model and tokenizer: {args.model}")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set pad token to eos token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Load and clean data
        print(f"Loading data from {args.data_dir}")
        samples = load_and_clean_data(args.data_dir)
        print(f"Loaded {len(samples)} writing samples")
        
        if len(samples) == 0:
            print("Error: No valid text samples found. Please add your writing samples as .txt files in the data directory.")
            return
        
        # Prepare datasets
        print("Preparing datasets...")
        tokenized_dataset = prepare_datasets(samples, tokenizer, args.max_length, args.val_split)
        
        # Train model
        train_model(args, tokenized_dataset, tokenizer)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()