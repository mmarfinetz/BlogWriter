#!/usr/bin/env python3

import argparse
import os
import sys
from anthropic import Anthropic
import tiktoken

def parse_args():
    parser = argparse.ArgumentParser(description="Generate blog posts with Anthropic's Claude 3.5")
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        required=True, 
        help="Path to a .txt file containing the prompt"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="claude-3-5-sonnet-20240620", 
        help="Anthropic model to use (default: claude-3-5-sonnet-20240620)"
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
        default=4000, 
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--max_input_tokens", 
        type=int, 
        default=100000, 
        help="Maximum number of tokens in the input prompt"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_blog.txt", 
        help="Output file for the generated text"
    )
    
    return parser.parse_args()

def count_tokens(text):
    """Approximately count tokens for a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")  # Close enough approximation for Claude
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text) // 4

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if ANTHROPIC_API_KEY is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set your Anthropic API key with:")
        print("export ANTHROPIC_API_KEY=your-api-key")
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
    
    # Check prompt length
    prompt_tokens = count_tokens(prompt)
    print(f"Estimated prompt length: {prompt_tokens} tokens")
    
    if prompt_tokens > args.max_input_tokens:
        print(f"Warning: Prompt is too long ({prompt_tokens} tokens).")
        print(f"It will be truncated to approximately {args.max_input_tokens} tokens.")
        
        # Truncate the prompt to max_input_tokens
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(prompt)
        truncated_tokens = tokens[:args.max_input_tokens]
        prompt = encoding.decode(truncated_tokens)
    
    print(f"\nGenerating blog post with the following parameters:")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output file: {args.output}")
    print("\nGenerating...")
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Generate text
    try:
        response = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system="You are an expert blogger who writes insightful, analytical content on complex topics. Your writing style is clear, technical yet accessible.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_text = response.content[0].text
        
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