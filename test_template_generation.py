#!/usr/bin/env python3
"""
Test script for template-based generation with the improved BlogGenerator.
"""

import os
import sys
import argparse
from src.generator import BlogGenerator
from src.templates import ContentType, StyleType, ToneType, FormatType

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test template-based content generation")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model or adapter")
    
    parser.add_argument("--topic", type=str, required=True,
                       help="Topic for content generation")
    
    parser.add_argument("--content_type", type=str, default="blog",
                       choices=[ct.value for ct in ContentType],
                       help="Type of content to generate")
    
    parser.add_argument("--style", type=str, default=None,
                       choices=[st.value for st in StyleType],
                       help="Writing style to use")
    
    parser.add_argument("--tone", type=str, default=None,
                       choices=[tt.value for tt in ToneType],
                       help="Tone of the content")
    
    parser.add_argument("--format", type=str, default=None,
                       choices=[ft.value for ft in FormatType],
                       help="Format structure for the content")
    
    parser.add_argument("--max_length", type=int, default=1500,
                       help="Maximum length of generated text")
    
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more creative)")
    
    parser.add_argument("--use_examples", action="store_true",
                       help="Use few-shot examples from data directory")
    
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing example content")
    
    parser.add_argument("--output_file", type=str, default=None,
                       help="File to save the generated content")
    
    parser.add_argument("--maintain_consistency", action="store_true",
                       help="Apply consistency enforcement throughout content")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"Initializing generator with model: {args.model_path}")
    generator = BlogGenerator(
        model_path=args.model_path,
        style_aware=True,
        templates_dir="prompts",
        data_dir=args.data_dir if args.use_examples else None
    )
    
    print(f"Generating {args.content_type} about '{args.topic}'...")
    print(f"Style: {args.style or 'default'}, Tone: {args.tone or 'default'}, Format: {args.format or 'default'}")
    
    generated_texts = generator.generate_with_template(
        topic=args.topic,
        content_type=args.content_type,
        style=args.style,
        tone=args.tone,
        format_type=args.format,
        max_length=args.max_length,
        temperature=args.temperature,
        use_examples=args.use_examples,
        maintain_consistency=args.maintain_consistency
    )
    
    print("\nGeneration complete! Result:")
    print("=" * 80)
    print(generated_texts[0])
    print("=" * 80)
    
    if args.output_file:
        generator.save_generated_text(generated_texts[0], args.output_file)
        print(f"Content saved to {args.output_file}")

if __name__ == "__main__":
    main()