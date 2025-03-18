#!/usr/bin/env python3

import argparse
import os
from src.generator import BlogGenerator
from src.evaluation import TextEvaluator

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
        "--style_aware", 
        action="store_true", 
        help="Enable style-aware generation using learned style metrics"
    )
    parser.add_argument(
        "--style_strength", 
        type=float, 
        default=0.8, 
        help="How strongly to enforce style (0-1, higher means stronger enforcement)"
    )
    parser.add_argument(
        "--style_summary", 
        action="store_true", 
        help="Display a summary of the detected writing style metrics"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_blog.txt", 
        help="Output file for the generated text"
    )
    
    # Evaluation options
    parser.add_argument(
        "--evaluate", 
        action="store_true", 
        help="Evaluate the generated content"
    )
    parser.add_argument(
        "--references", 
        type=str, 
        default=None, 
        help="Directory containing reference texts for evaluation comparison"
    )
    parser.add_argument(
        "--eval_output", 
        type=str, 
        default="evaluation_report.txt", 
        help="Output file for the evaluation report"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Generate evaluation visualizations"
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
    generator = BlogGenerator(model_path=args.model, style_aware=args.style_aware)
    
    # If style summary is requested, display it
    if args.style_summary:
        style_info = generator.get_style_summary()
        print("\nWriting Style Analysis:")
        print("=" * 50)
        
        if "error" in style_info:
            print(f"Error: {style_info['error']}")
        else:
            print(f"Lexical Diversity: {style_info['lexical_diversity']:.3f}")
            print(f"Average Sentence Length: {style_info['avg_sentence_length']:.1f} words")
            
            print("\nSentence Distribution:")
            total = sum(style_info['sentence_distribution'].values())
            if total > 0:
                print(f"  Short sentences: {style_info['sentence_distribution']['short']/total*100:.1f}%")
                print(f"  Medium sentences: {style_info['sentence_distribution']['medium']/total*100:.1f}%")
                print(f"  Long sentences: {style_info['sentence_distribution']['long']/total*100:.1f}%")
            
            print("\nTop Transition Phrases:")
            for phrase in style_info.get('top_transitions', []):
                print(f"  - {phrase}")
            
            print(f"\nPassive Voice Usage: {style_info['passive_voice_ratio']*100:.1f}%")
            print(f"Adverb Usage: {style_info['adverb_ratio']*100:.1f}%")
        
        print("=" * 50)
    
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
    if args.style_aware:
        print(f"Style-aware generation: Enabled (strength: {args.style_strength})")
    else:
        print(f"Style-aware generation: Disabled")
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
        num_return_sequences=1,
        style_strength=args.style_strength if args.style_aware else 0
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
    
    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating generated content...")
        
        # Initialize TextEvaluator
        evaluator = TextEvaluator(use_grammar_check=True, use_bert_score=False)
        
        # Load reference texts if provided
        reference_texts = []
        if args.references and os.path.isdir(args.references):
            for filename in os.listdir(args.references):
                if filename.endswith(".txt"):
                    try:
                        with open(os.path.join(args.references, filename), "r", encoding="utf-8") as f:
                            content = f.read()
                            reference_texts.append(content)
                    except Exception as e:
                        print(f"Warning: Could not read reference file {filename}: {e}")
            print(f"Loaded {len(reference_texts)} reference texts for comparison")
        
        # Evaluate the generated text
        metrics = evaluator.evaluate_text(
            generated_text=generated_texts[0],
            reference_texts=reference_texts if reference_texts else None,
            style_metrics=generator.style_metrics if args.style_aware else None
        )
        
        # Calculate perplexity
        try:
            perplexity = evaluator.calculate_perplexity(
                text=generated_texts[0],
                model_name="gpt2"  # Using gpt2 as a standard baseline
            )
            metrics["perplexity"] = perplexity
        except Exception as e:
            print(f"Warning: Could not calculate perplexity: {e}")
        
        # Format and save evaluation report
        report = TextEvaluator.format_metrics(metrics, include_descriptions=True)
        
        with open(args.eval_output, "w", encoding="utf-8") as f:
            f.write(report)
        
        # Print key metrics
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"Word count: {metrics.get('word_count', 0)}")
        print(f"Lexical diversity: {metrics.get('lexical_diversity', 0):.3f}")
        print(f"Flesch reading ease: {metrics.get('flesch_reading_ease', 0):.1f}")
        if 'perplexity' in metrics:
            print(f"Perplexity: {metrics.get('perplexity', 0):.2f}")
        
        if reference_texts:
            print(f"ROUGE-1 F1: {metrics.get('rouge1_f1', 0):.4f}")
            print(f"BLEU-4: {metrics.get('bleu4', 0):.4f}")
            print(f"Content similarity: {metrics.get('content_similarity', 0):.4f}")
        
        if args.style_aware:
            print(f"Style similarity: {metrics.get('overall_style_similarity', 0):.4f}")
        
        # Generate visualizations if requested
        if args.visualize:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the visualization
            plt.figure(figsize=(10, 8))
            
            # Plot key metrics
            metrics_to_plot = {
                "Lexical Diversity": metrics.get("lexical_diversity", 0),
                "Readability": metrics.get("flesch_reading_ease", 0) / 100,  # Normalize to 0-1
                "Grammar": 1 - min(1, metrics.get("grammar_error_density", 0) / 10)  # Invert errors
            }
            
            if reference_texts:
                metrics_to_plot["ROUGE-1"] = metrics.get("rouge1_f1", 0)
                metrics_to_plot["BLEU-4"] = metrics.get("bleu4", 0)
                metrics_to_plot["Content Similarity"] = metrics.get("content_similarity", 0)
            
            plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=sns.color_palette("muted"))
            plt.ylim(0, 1)
            plt.title("Content Quality Metrics")
            
            # Save visualization
            viz_path = os.path.splitext(args.eval_output)[0] + "_visualization.png"
            plt.savefig(viz_path)
            
            print(f"\nEvaluation visualization saved to {viz_path}")
        
        print(f"\nDetailed evaluation report saved to {args.eval_output}")

if __name__ == "__main__":
    main()