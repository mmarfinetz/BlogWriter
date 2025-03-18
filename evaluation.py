#!/usr/bin/env python3
"""
Standalone evaluation script for BlogWriter's generated content.

This script provides a command-line interface to evaluate text files, compare them
with reference samples, and generate detailed evaluation reports with quality metrics.
"""

import argparse
import os
import json
import sys
from typing import Dict, Any, Optional, List
from src.evaluation import BlogEvaluator, TextEvaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated blog content quality using NLP metrics"
    )
    
    # Input arguments
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input", type=str, required=False,
        help="Path to a single text file to evaluate"
    )
    input_group.add_argument(
        "--input_dir", type=str, required=False,
        help="Directory of text files to evaluate in batch"
    )
    input_group.add_argument(
        "--references", type=str, required=False,
        help="Directory containing reference text files for comparison"
    )
    input_group.add_argument(
        "--style_metrics", type=str, required=False,
        help="Path to JSON file with target style metrics for comparison"
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_dir", type=str, default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    output_group.add_argument(
        "--report_file", type=str, required=False,
        help="Filename for the evaluation report (default: derived from input filename)"
    )
    output_group.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization charts of evaluation metrics"
    )
    output_group.add_argument(
        "--format", type=str, choices=["text", "json"], default="text",
        help="Output format for evaluation results"
    )
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--metrics", type=str, default="all",
        help="Comma-separated list of metrics to include (default: all)"
    )
    eval_group.add_argument(
        "--grammar_check", action="store_true",
        help="Enable grammar checking (slower but more comprehensive)"
    )
    eval_group.add_argument(
        "--bert_score", action="store_true",
        help="Use BERTScore for semantic similarity (requires GPU for speed)"
    )
    eval_group.add_argument(
        "--perplexity", action="store_true",
        help="Calculate perplexity using a language model"
    )
    eval_group.add_argument(
        "--perplexity_model", type=str, default="gpt2",
        help="Model to use for perplexity calculation (default: gpt2)"
    )
    eval_group.add_argument(
        "--descriptions", action="store_true",
        help="Include metric descriptions in the text report"
    )
    
    # Comparison arguments
    comparison_group = parser.add_argument_group("Comparison Options")
    comparison_group.add_argument(
        "--compare", type=str, required=False,
        help="Path to another text file to compare with the input"
    )
    comparison_group.add_argument(
        "--compare_dir", type=str, required=False,
        help="Directory of files to compare with input_dir (must have matching filenames)"
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments."""
    # Ensure we have at least one input source
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input_dir must be specified")
        return False
    
    # If compare is specified, input must be a single file
    if args.compare and not args.input:
        print("Error: --compare requires --input to be specified")
        return False
    
    # If compare_dir is specified, input_dir must be specified
    if args.compare_dir and not args.input_dir:
        print("Error: --compare_dir requires --input_dir to be specified")
        return False
    
    # Check that input files/directories exist
    if args.input and not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return False
    
    if args.input_dir and not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return False
    
    if args.references and not os.path.isdir(args.references):
        print(f"Error: References directory not found: {args.references}")
        return False
    
    if args.style_metrics and not os.path.isfile(args.style_metrics):
        print(f"Error: Style metrics file not found: {args.style_metrics}")
        return False
    
    if args.compare and not os.path.isfile(args.compare):
        print(f"Error: Comparison file not found: {args.compare}")
        return False
    
    if args.compare_dir and not os.path.isdir(args.compare_dir):
        print(f"Error: Comparison directory not found: {args.compare_dir}")
        return False
    
    return True

def evaluate_single_file(args):
    """Evaluate a single text file."""
    try:
        # Initialize evaluator
        evaluator = BlogEvaluator(
            output_dir=args.output_dir,
            reference_dir=args.references,
            style_metrics_path=args.style_metrics,
            use_grammar_check=args.grammar_check,
            use_bert_score=args.bert_score
        )
        
        # Read input text
        with open(args.input, "r", encoding="utf-8") as f:
            generated_text = f.read()
        
        # Read comparison text if provided
        comparison_text = None
        if args.compare:
            with open(args.compare, "r", encoding="utf-8") as f:
                comparison_text = f.read()
        
        # Run evaluation
        metrics = evaluator.evaluate(
            generated_text=generated_text,
            run_all_metrics=args.metrics == "all"
        )
        
        # Calculate perplexity if requested
        if args.perplexity:
            perplexity = evaluator.evaluator.calculate_perplexity(
                text=generated_text,
                model_name=args.perplexity_model
            )
            metrics["perplexity"] = perplexity
        
        # Compare with another text if requested
        if comparison_text:
            comparison_metrics = evaluator.evaluator.compare_with_references(
                generated_text=generated_text,
                reference_texts=[comparison_text]
            )
            metrics.update({f"direct_comparison_{k}": v for k, v in comparison_metrics.items()})
        
        # Determine report filename
        if args.report_file:
            report_path = os.path.join(args.output_dir, args.report_file)
        else:
            input_basename = os.path.basename(args.input)
            report_path = os.path.join(args.output_dir, f"{os.path.splitext(input_basename)[0]}_evaluation")
            
            # Add extension based on format
            if args.format == "json":
                report_path += ".json"
            else:
                report_path += ".txt"
        
        # Generate and save report
        if args.format == "json":
            # Save JSON report
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            # Generate text report
            report = evaluator.generate_report(
                metrics=metrics,
                output_file=report_path,
                include_descriptions=args.descriptions,
                include_visualizations=args.visualize
            )
            print("\nEvaluation Summary:")
            print("=" * 50)
            
            # Print key metrics to console
            summary_metrics = [
                ("Word count", metrics.get("word_count", 0)),
                ("Lexical diversity", metrics.get("lexical_diversity", 0)),
                ("Flesch reading ease", metrics.get("flesch_reading_ease", 0)),
                ("Coleman-Liau index", metrics.get("coleman_liau_index", 0))
            ]
            
            if "grammar_error_count" in metrics:
                summary_metrics.append(("Grammar errors", metrics.get("grammar_error_count", 0)))
            
            if "rouge1_f1" in metrics:
                summary_metrics.append(("ROUGE-1 F1", metrics.get("rouge1_f1", 0)))
                summary_metrics.append(("BLEU-4", metrics.get("bleu4", 0)))
            
            if "overall_style_similarity" in metrics:
                summary_metrics.append(("Style similarity", metrics.get("overall_style_similarity", 0)))
            
            if "perplexity" in metrics:
                summary_metrics.append(("Perplexity", metrics.get("perplexity", 0)))
            
            for name, value in summary_metrics:
                if isinstance(value, float):
                    print(f"{name}: {value:.4f}")
                else:
                    print(f"{name}: {value}")
            
        print(f"\nFull evaluation report saved to: {report_path}")
        
        if args.visualize:
            print(f"Visualizations saved to: {os.path.splitext(report_path)[0]}_visualizations.png")
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating file: {e}")
        return None

def batch_evaluate_files(args):
    """Evaluate multiple text files in a directory."""
    try:
        # Initialize evaluator
        evaluator = BlogEvaluator(
            output_dir=args.output_dir,
            reference_dir=args.references,
            style_metrics_path=args.style_metrics,
            use_grammar_check=args.grammar_check,
            use_bert_score=args.bert_score
        )
        
        # Collect input files
        input_files = {}
        for filename in os.listdir(args.input_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(args.input_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        input_files[os.path.splitext(filename)[0]] = content
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
        
        if not input_files:
            print(f"No text files found in {args.input_dir}")
            return None
        
        print(f"Found {len(input_files)} text files to evaluate")
        
        # Collect comparison files if requested
        comparison_files = {}
        if args.compare_dir:
            for name in input_files.keys():
                filename = f"{name}.txt"
                file_path = os.path.join(args.compare_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            comparison_files[name] = [content]
                    except Exception as e:
                        print(f"Warning: Could not read comparison file {filename}: {e}")
        
        # Evaluate all files
        results = evaluator.batch_evaluate(
            generated_texts=input_files,
            reference_texts=comparison_files if comparison_files else None,
            output_dir=args.output_dir,
            include_visualizations=args.visualize
        )
        
        # Calculate perplexity if requested
        if args.perplexity:
            for name, text in input_files.items():
                try:
                    perplexity = evaluator.evaluator.calculate_perplexity(
                        text=text,
                        model_name=args.perplexity_model
                    )
                    results[name]["perplexity"] = perplexity
                except Exception as e:
                    print(f"Warning: Could not calculate perplexity for {name}: {e}")
        
        # Save batch results in JSON format
        batch_results_path = os.path.join(args.output_dir, "batch_evaluation_results.json")
        with open(batch_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch evaluation completed.")
        print(f"Individual reports saved to {args.output_dir}")
        print(f"Batch results summary saved to {batch_results_path}")
        print(f"Comparison visualization saved to {os.path.join(args.output_dir, 'comparison_visualization.png')}")
        
        return results
    
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        return None

def main():
    """Main entry point for the evaluation script."""
    args = parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    if args.input:
        # Single file evaluation
        evaluate_single_file(args)
    elif args.input_dir:
        # Batch evaluation
        batch_evaluate_files(args)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()