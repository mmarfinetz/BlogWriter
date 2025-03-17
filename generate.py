#!/usr/bin/env python3

import argparse
import os
from src.generator import BlogGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Generate blog posts with your fine-tuned model")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="", 
        help="Starting text for generation"
    )
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default="", 
        help="Path to a .txt file containing the prompt"
    )
    parser.add_argument(
        "--length", 
        type=int, 
        default=1000, 
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Sampling temperature (higher = more creative, lower = more focused)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.2, 
        help="Penalty for repetition"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_blog.txt", 
        help="Output file for the generated text"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' doesn't exist")
        return
    
    # Initialize generator
    generator = BlogGenerator(model_path=args.model)
    
    # Get prompt
    prompt = args.prompt
    
    # Check if prompt file is provided
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            try:
                with open(args.prompt_file, 'r', encoding='utf-8') as file:
                    prompt = file.read()
                print(f"Loaded prompt from file: {args.prompt_file}")
            except Exception as e:
                print(f"Error reading prompt file: {e}")
                return
        else:
            print(f"Error: Prompt file '{args.prompt_file}' doesn't exist")
            return
    
    # If neither prompt nor prompt_file is provided, ask for input
    if not prompt and not args.prompt_file:
        prompt = input("Enter a prompt for your blog post: ")
    
    print(f"\nGenerating blog post with the following parameters:")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.length}")
    print(f"Output file: {args.output}")
    print("\nGenerating...")
    
    # Generate text
    generated_texts = generator.generate(
        prompt=prompt,
        max_length=args.length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1
    )
    
    # Save generated text
    generator.save_generated_text(generated_texts[0], args.output)
    
    # Print a preview
    preview_length = min(500, len(generated_texts[0]))
    print(f"\nPreview of generated blog post:")
    print("=" * 50)
    print(generated_texts[0][:preview_length] + "..." if len(generated_texts[0]) > preview_length else generated_texts[0])
    print("=" * 50)
    print(f"\nFull text saved to {args.output}")

if __name__ == "__main__":
    main()