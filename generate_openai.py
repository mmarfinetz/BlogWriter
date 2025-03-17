#!/usr/bin/env python3

import argparse
import os
import sys
import openai
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="Generate blog posts with OpenAI's GPT-4o mini")
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        required=True, 
        help="Path to a .txt file containing the prompt"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini", 
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature (higher = more creative, lower = more focused)"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2000, 
        help="Maximum number of tokens to generate"
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
    
    # Check if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key with:")
        print("export OPENAI_API_KEY=your-api-key")
        return
    
    # Check if prompt file exists
    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file '{args.prompt_file}' doesn't exist")
        return
    
    # Read prompt from file
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()
        print(f"Loaded prompt from file: {args.prompt_file}")
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return
    
    print(f"\nGenerating blog post with the following parameters:")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output file: {args.output}")
    print("\nGenerating...")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Generate text
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are an expert blogger who writes insightful, analytical content on complex topics. Your writing style is clear, technical yet accessible."},
                {"role": "user", "content": prompt}
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        generated_text = response.choices[0].message.content
        
        # Save generated text
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(generated_text)
        
        # Print a preview
        preview_length = min(500, len(generated_text))
        print(f"\nPreview of generated blog post:")
        print("=" * 50)
        print(generated_text[:preview_length] + "..." if len(generated_text) > preview_length else generated_text)
        print("=" * 50)
        print(f"\nFull text saved to {args.output}")
        
    except Exception as e:
        print(f"Error generating text: {e}")
        return

if __name__ == "__main__":
    main() 