#!/usr/bin/env python3

import argparse
import os
from src.trainer import BlogModelTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom language model for blog generation")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2", 
        help="Base model to fine-tune (e.g., gpt2, gpt2-medium, distilgpt2)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data", 
        help="Directory containing your writing samples"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models", 
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=8, 
        help="Gradient accumulation steps"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print training configuration
    print(f"Training configuration:")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.grad_accum}")
    
    # Check if data directory contains files
    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' is empty or doesn't exist")
        print("Please add your writing samples (as .txt files) to the data directory")
        return
    
    # Initialize and train the model
    trainer = BlogModelTrainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Train the model
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum
    )
    
    print("\nTraining complete! Your fine-tuned model is ready for generating blog posts.")
    print(f"To generate content, run: python generate.py --model {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    main()