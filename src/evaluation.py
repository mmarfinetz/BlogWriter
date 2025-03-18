"""
Evaluation module for BlogWriter's output quality.

This module provides comprehensive metrics and evaluation tools to assess
the quality of generated blog content, comparing it with reference texts and
analyzing various text quality aspects.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import sacrebleu
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import language_tool_python
from collections import Counter

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextEvaluator:
    """Evaluates the quality of generated text using various NLP metrics."""
    
    def __init__(
        self,
        use_grammar_check: bool = True,
        use_bert_score: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text evaluator.
        
        Args:
            use_grammar_check: Whether to use LanguageTool for grammar checking (can be slow)
            use_bert_score: Whether to use BERTScore (requires additional dependencies)
            cache_dir: Directory to cache models (e.g., for BERTScore)
        """
        self.use_grammar_check = use_grammar_check
        self.use_bert_score = use_bert_score
        self.cache_dir = cache_dir
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method4
        
        # Initialize grammar checker if enabled
        if self.use_grammar_check:
            try:
                self.grammar_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                print(f"Warning: Could not initialize LanguageTool: {e}")
                print("Grammar checking will be disabled.")
                self.use_grammar_check = False
        
        # Initialize BERTScore if enabled
        if self.use_bert_score:
            try:
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(
                    lang="en", 
                    rescale_with_baseline=True,
                    cache_dir=self.cache_dir
                )
            except ImportError:
                print("Warning: BERTScore not available. Install with 'pip install bert-score'")
                self.use_bert_score = False
    
    def evaluate_text(
        self, 
        generated_text: str, 
        reference_texts: Optional[List[str]] = None,
        style_metrics: Optional[Dict[str, Any]] = None,
        run_all_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensively evaluate generated text quality.
        
        Args:
            generated_text: Text to evaluate
            reference_texts: List of reference texts to compare against
            style_metrics: Dictionary of target style metrics for comparison
            run_all_metrics: Whether to run all available metrics (slower)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        metrics = self.calculate_basic_metrics(generated_text)
        
        # Readability metrics
        metrics.update(self.calculate_readability_metrics(generated_text))
        
        # Style metrics
        style_analysis = self.analyze_style(generated_text)
        metrics.update(style_analysis)
        
        # Grammar and language quality if enabled
        if self.use_grammar_check or run_all_metrics:
            grammar_metrics = self.check_grammar(generated_text)
            metrics.update(grammar_metrics)
        
        # If reference texts are provided, calculate comparison metrics
        if reference_texts and len(reference_texts) > 0:
            comparison_metrics = self.compare_with_references(
                generated_text, reference_texts
            )
            metrics.update(comparison_metrics)
            
            # Style comparison if style metrics are provided
            if style_metrics:
                style_comparison = self.compare_style(style_analysis, style_metrics)
                metrics.update(style_comparison)
        
        return metrics
    
    def calculate_basic_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate basic text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of basic metrics
        """
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Filter out punctuation for word count
        words_no_punct = [word for word in words if word.isalnum()]
        
        # Calculate metrics
        word_count = len(words_no_punct)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Lexical diversity (unique words / total words)
        unique_words = len(set(w.lower() for w in words_no_punct))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Character count and average word length
        char_count = len(text)
        avg_word_length = sum(len(word) for word in words_no_punct) / word_count if word_count > 0 else 0
        
        # Paragraphs
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
            "character_count": char_count
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate various readability scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of readability metrics
        """
        # Ensure text is long enough for meaningful analysis
        if len(text.split()) < 10:
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "smog_index": 0,
                "coleman_liau_index": 0,
                "automated_readability_index": 0,
                "dale_chall_readability_score": 0
            }
            
        try:
            # Calculate various readability metrics using textstat
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "smog_index": textstat.smog_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(text)
            }
        except Exception as e:
            print(f"Warning: Error calculating readability metrics: {e}")
            return {
                "readability_error": str(e)
            }
    
    def analyze_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze writing style characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of style metrics
        """
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Calculate sentence lengths
        sentence_lengths = []
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence)
            words_no_punct = [word for word in words_in_sentence if word.isalnum()]
            sentence_lengths.append(len(words_no_punct))
        
        # Calculate average sentence length
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Categorize sentences by length
        short_sentences = len([l for l in sentence_lengths if l <= 10])
        medium_sentences = len([l for l in sentence_lengths if 10 < l <= 20])
        long_sentences = len([l for l in sentence_lengths if l > 20])
        sentence_categories = {
            "short_sentences": short_sentences / len(sentences) if sentences else 0,
            "medium_sentences": medium_sentences / len(sentences) if sentences else 0,
            "long_sentences": long_sentences / len(sentences) if sentences else 0
        }
        
        # POS tagging for deeper analysis
        pos_tags = nltk.pos_tag(words)
        
        # Count adverbs (RB, RBR, RBS tags)
        adverbs = [word for word, tag in pos_tags if tag.startswith('RB')]
        adverb_ratio = len(adverbs) / len(words) if words else 0
        
        # Count adjectives (JJ, JJR, JJS tags)
        adjectives = [word for word, tag in pos_tags if tag.startswith('JJ')]
        adjective_ratio = len(adjectives) / len(words) if words else 0
        
        # Calculate passive voice indicators (rough approximation)
        passive_indicators = ['was', 'were', 'been', 'be', 'is', 'are']
        passive_count = 0
        
        for i, (word, tag) in enumerate(pos_tags):
            if word.lower() in passive_indicators and i < len(pos_tags) - 1:
                # Check for past participle (VBN) after passive indicator
                if pos_tags[i+1][1] == 'VBN':
                    passive_count += 1
        
        passive_ratio = passive_count / len(sentences) if sentences else 0
        
        # Analyze transition phrases
        transition_phrases = [
            'however', 'therefore', 'furthermore', 'nevertheless', 'meanwhile',
            'consequently', 'in addition', 'for example', 'as a result',
            'on the other hand', 'in conclusion', 'in summary', 'for instance',
            'in contrast', 'similarly', 'in other words', 'in fact'
        ]
        
        transition_counts = {}
        for phrase in transition_phrases:
            count = text.lower().count(phrase)
            if count > 0:
                transition_counts[phrase] = count
        
        # Sort by frequency and convert to ordered dict
        top_transitions = dict(sorted(transition_counts.items(), 
                                      key=lambda item: item[1], 
                                      reverse=True)[:5])
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
            "sentence_categories": sentence_categories,
            "adverb_ratio": adverb_ratio,
            "adjective_ratio": adjective_ratio,
            "passive_voice_ratio": passive_ratio,
            "top_transition_phrases": top_transitions
        }
    
    def check_grammar(self, text: str) -> Dict[str, Any]:
        """
        Check text for grammatical errors.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of grammar metrics
        """
        if not self.use_grammar_check:
            return {"grammar_check_enabled": False}
        
        try:
            # Check grammar with LanguageTool
            matches = self.grammar_tool.check(text)
            
            # Count errors by category
            error_categories = Counter()
            for match in matches:
                error_categories[match.category] = error_categories.get(match.category, 0) + 1
            
            # Calculate error density (errors per 100 words)
            word_count = len(word_tokenize(text))
            error_density = len(matches) / (word_count / 100) if word_count > 0 else 0
            
            return {
                "grammar_error_count": len(matches),
                "grammar_error_density": error_density,
                "grammar_error_categories": dict(error_categories)
            }
        except Exception as e:
            print(f"Warning: Error during grammar checking: {e}")
            return {"grammar_check_error": str(e)}
    
    def compare_with_references(
        self, 
        generated_text: str, 
        reference_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Compare generated text with reference texts using various metrics.
        
        Args:
            generated_text: Text to evaluate
            reference_texts: List of reference texts to compare against
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison_metrics = {}
        
        # Calculate ROUGE scores
        rouge_scores = {metric: [] for metric in ['rouge1', 'rouge2', 'rougeL']}
        
        for reference in reference_texts:
            # Skip empty references
            if not reference.strip():
                continue
                
            scores = self.rouge_scorer.score(reference, generated_text)
            
            for metric in rouge_scores.keys():
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average ROUGE scores across all references
        for metric in rouge_scores.keys():
            if rouge_scores[metric]:
                comparison_metrics[f"{metric}_f1"] = np.mean(rouge_scores[metric])
            else:
                comparison_metrics[f"{metric}_f1"] = 0
        
        # Calculate BLEU scores
        try:
            # Tokenize texts for BLEU calculation
            reference_tokens = [word_tokenize(ref.lower()) for ref in reference_texts]
            generated_tokens = word_tokenize(generated_text.lower())
            
            # Calculate BLEU-1 to BLEU-4 scores
            bleu1 = sentence_bleu(reference_tokens, generated_tokens, 
                                   weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            bleu2 = sentence_bleu(reference_tokens, generated_tokens, 
                                   weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            bleu3 = sentence_bleu(reference_tokens, generated_tokens, 
                                   weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            bleu4 = sentence_bleu(reference_tokens, generated_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            
            comparison_metrics.update({
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4
            })
            
            # Calculate SacreBLEU for a more standard implementation
            try:
                sacrebleu_score = sacrebleu.corpus_bleu(
                    [generated_text], 
                    [[ref] for ref in reference_texts]
                ).score / 100  # Normalize to 0-1 scale
                comparison_metrics["sacrebleu"] = sacrebleu_score
            except Exception as e:
                print(f"Warning: Error calculating SacreBLEU: {e}")
        
        except Exception as e:
            print(f"Warning: Error calculating BLEU scores: {e}")
            comparison_metrics["bleu_error"] = str(e)
        
        # Calculate content similarity using TF-IDF and cosine similarity
        try:
            # Ensure we have non-empty texts
            valid_texts = [text for text in [generated_text] + reference_texts if text.strip()]
            if len(valid_texts) < 2:
                comparison_metrics["content_similarity"] = 0
            else:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                
                # Calculate cosine similarity between generated text and each reference
                similarities = []
                for i in range(1, len(valid_texts)):
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[i:i+1])[0][0]
                    similarities.append(similarity)
                
                # Use the highest similarity as the score
                comparison_metrics["content_similarity"] = max(similarities) if similarities else 0
        except Exception as e:
            print(f"Warning: Error calculating content similarity: {e}")
            comparison_metrics["content_similarity_error"] = str(e)
        
        # Calculate BERTScore if enabled
        if self.use_bert_score:
            try:
                # Calculate BERTScore
                p, r, f1 = self.bert_scorer.score([generated_text], reference_texts)
                comparison_metrics.update({
                    "bert_score_precision": p.mean().item(),
                    "bert_score_recall": r.mean().item(),
                    "bert_score_f1": f1.mean().item()
                })
            except Exception as e:
                print(f"Warning: Error calculating BERTScore: {e}")
                comparison_metrics["bert_score_error"] = str(e)
        
        return comparison_metrics
    
    def compare_style(
        self, 
        generated_style: Dict[str, Any], 
        target_style: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare generated text style with target style metrics.
        
        Args:
            generated_style: Style metrics for generated text
            target_style: Target style metrics to compare against
            
        Returns:
            Dictionary of style similarity metrics
        """
        style_similarity = {}
        
        # Compare sentence length
        if "avg_sentence_length" in generated_style and "avg_sentence_length" in target_style:
            gen_len = generated_style["avg_sentence_length"]
            target_len = target_style["avg_sentence_length"]
            
            # Calculate similarity as 1 - normalized difference
            max_diff = 20  # Maximum expected difference
            len_similarity = max(0, 1 - (abs(gen_len - target_len) / max_diff))
            style_similarity["sentence_length_similarity"] = len_similarity
        
        # Compare sentence categories
        if "sentence_categories" in generated_style and "sentence_categories" in target_style:
            gen_cats = generated_style["sentence_categories"]
            target_cats = target_style["sentence_categories"]
            
            cat_similarities = []
            for cat in ["short_sentences", "medium_sentences", "long_sentences"]:
                if cat in gen_cats and cat in target_cats:
                    similarity = 1 - abs(gen_cats[cat] - target_cats[cat])
                    cat_similarities.append(similarity)
            
            if cat_similarities:
                style_similarity["sentence_distribution_similarity"] = np.mean(cat_similarities)
        
        # Compare passive voice ratio
        if "passive_voice_ratio" in generated_style and "passive_voice_ratio" in target_style:
            gen_passive = generated_style["passive_voice_ratio"]
            target_passive = target_style["passive_voice_ratio"]
            
            passive_similarity = 1 - abs(gen_passive - target_passive)
            style_similarity["passive_voice_similarity"] = passive_similarity
        
        # Compare adverb and adjective usage
        for feature in ["adverb_ratio", "adjective_ratio"]:
            if feature in generated_style and feature in target_style:
                gen_value = generated_style[feature]
                target_value = target_style[feature]
                
                similarity = 1 - abs(gen_value - target_value)
                style_similarity[f"{feature}_similarity"] = similarity
        
        # Compare transition phrase usage
        if "top_transition_phrases" in generated_style and "top_transition_phrases" in target_style:
            gen_trans = set(generated_style["top_transition_phrases"].keys())
            target_trans = set(target_style["top_transition_phrases"].keys())
            
            # Calculate Jaccard similarity
            if target_trans:
                common = len(gen_trans.intersection(target_trans))
                union = len(gen_trans.union(target_trans))
                style_similarity["transition_phrase_similarity"] = common / union if union > 0 else 0
        
        # Calculate overall style similarity as weighted average
        if style_similarity:
            weights = {
                "sentence_length_similarity": 0.3,
                "sentence_distribution_similarity": 0.2,
                "passive_voice_similarity": 0.1,
                "adverb_ratio_similarity": 0.1,
                "adjective_ratio_similarity": 0.1,
                "transition_phrase_similarity": 0.2
            }
            
            weighted_sum = 0
            weight_sum = 0
            
            for metric, value in style_similarity.items():
                if metric in weights:
                    weighted_sum += value * weights[metric]
                    weight_sum += weights[metric]
            
            style_similarity["overall_style_similarity"] = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        return style_similarity
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any], include_descriptions: bool = True) -> str:
        """
        Format metrics dictionary as a human-readable string.
        
        Args:
            metrics: Dictionary of evaluation metrics
            include_descriptions: Whether to include descriptions of metrics
            
        Returns:
            Formatted string with metrics and descriptions
        """
        # Dictionary of metric descriptions
        descriptions = {
            # Basic metrics
            "word_count": "Total number of words in the text",
            "sentence_count": "Total number of sentences in the text",
            "paragraph_count": "Total number of paragraphs in the text",
            "avg_words_per_sentence": "Average number of words per sentence",
            "avg_word_length": "Average length of words in characters",
            "lexical_diversity": "Ratio of unique words to total words (0-1)",
            "character_count": "Total number of characters",
            
            # Readability metrics
            "flesch_reading_ease": "Flesch Reading Ease score (higher is easier to read, 0-100)",
            "flesch_kincaid_grade": "Flesch-Kincaid Grade Level (estimated school grade level)",
            "smog_index": "SMOG Index (estimated school grade level)",
            "coleman_liau_index": "Coleman-Liau Index (estimated school grade level)",
            "automated_readability_index": "Automated Readability Index (estimated school grade level)",
            "dale_chall_readability_score": "Dale-Chall Readability score (lower is easier)",
            
            # Style metrics
            "avg_sentence_length": "Average number of words per sentence",
            "sentence_length_std": "Standard deviation of sentence length",
            "adverb_ratio": "Proportion of words that are adverbs (0-1)",
            "adjective_ratio": "Proportion of words that are adjectives (0-1)",
            "passive_voice_ratio": "Estimated proportion of sentences with passive voice (0-1)",
            
            # Grammar metrics
            "grammar_error_count": "Total number of grammar and spelling errors",
            "grammar_error_density": "Number of errors per 100 words",
            
            # Comparison metrics
            "rouge1_f1": "ROUGE-1 F1 score - unigram overlap with reference (0-1)",
            "rouge2_f1": "ROUGE-2 F1 score - bigram overlap with reference (0-1)",
            "rougeL_f1": "ROUGE-L F1 score - longest common subsequence with reference (0-1)",
            "bleu1": "BLEU-1 score - unigram precision with reference (0-1)",
            "bleu2": "BLEU-2 score - bigram precision with reference (0-1)",
            "bleu3": "BLEU-3 score - trigram precision with reference (0-1)",
            "bleu4": "BLEU-4 score - 4-gram precision with reference (0-1)",
            "sacrebleu": "SacreBLEU score - standardized BLEU implementation (0-1)",
            "content_similarity": "TF-IDF cosine similarity with reference (0-1)",
            "bert_score_f1": "BERTScore F1 - semantic similarity with reference using BERT (0-1)",
            
            # Style comparison metrics
            "sentence_length_similarity": "Similarity of sentence length to reference style (0-1)",
            "sentence_distribution_similarity": "Similarity of sentence length distribution (0-1)",
            "passive_voice_similarity": "Similarity of passive voice usage (0-1)",
            "adverb_ratio_similarity": "Similarity of adverb usage (0-1)",
            "adjective_ratio_similarity": "Similarity of adjective usage (0-1)",
            "transition_phrase_similarity": "Similarity of transition phrase usage (0-1)",
            "overall_style_similarity": "Overall weighted style similarity score (0-1)"
        }
        
        # Format metrics by category
        formatted_text = "EVALUATION METRICS\n" + "=" * 80 + "\n\n"
        
        # Basic metrics
        formatted_text += "TEXT STATISTICS:\n" + "-" * 40 + "\n"
        for metric in ["word_count", "sentence_count", "paragraph_count", "avg_words_per_sentence", 
                       "avg_word_length", "lexical_diversity", "character_count"]:
            if metric in metrics:
                value = metrics[metric]
                value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                formatted_text += f"{metric}: {value_str}"
                if include_descriptions and metric in descriptions:
                    formatted_text += f" ({descriptions[metric]})"
                formatted_text += "\n"
        formatted_text += "\n"
        
        # Readability metrics
        if any(m in metrics for m in ["flesch_reading_ease", "flesch_kincaid_grade"]):
            formatted_text += "READABILITY METRICS:\n" + "-" * 40 + "\n"
            for metric in ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index", 
                          "coleman_liau_index", "automated_readability_index", 
                          "dale_chall_readability_score"]:
                if metric in metrics:
                    value = metrics[metric]
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    formatted_text += f"{metric}: {value_str}"
                    if include_descriptions and metric in descriptions:
                        formatted_text += f" ({descriptions[metric]})"
                    formatted_text += "\n"
            formatted_text += "\n"
        
        # Style metrics
        if "avg_sentence_length" in metrics:
            formatted_text += "STYLE ANALYSIS:\n" + "-" * 40 + "\n"
            for metric in ["avg_sentence_length", "sentence_length_std", "adverb_ratio", 
                          "adjective_ratio", "passive_voice_ratio"]:
                if metric in metrics:
                    value = metrics[metric]
                    value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                    formatted_text += f"{metric}: {value_str}"
                    if include_descriptions and metric in descriptions:
                        formatted_text += f" ({descriptions[metric]})"
                    formatted_text += "\n"
            
            # Sentence categories
            if "sentence_categories" in metrics:
                formatted_text += "\nSentence length distribution:\n"
                for cat, ratio in metrics["sentence_categories"].items():
                    formatted_text += f"  {cat}: {ratio*100:.1f}%\n"
            
            # Top transition phrases
            if "top_transition_phrases" in metrics:
                formatted_text += "\nTop transition phrases:\n"
                for phrase, count in metrics["top_transition_phrases"].items():
                    formatted_text += f"  '{phrase}': {count} occurrences\n"
            formatted_text += "\n"
        
        # Grammar metrics
        if "grammar_error_count" in metrics:
            formatted_text += "GRAMMAR QUALITY:\n" + "-" * 40 + "\n"
            for metric in ["grammar_error_count", "grammar_error_density"]:
                if metric in metrics:
                    value = metrics[metric]
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    formatted_text += f"{metric}: {value_str}"
                    if include_descriptions and metric in descriptions:
                        formatted_text += f" ({descriptions[metric]})"
                    formatted_text += "\n"
            
            # Error categories
            if "grammar_error_categories" in metrics and metrics["grammar_error_categories"]:
                formatted_text += "\nGrammar error types:\n"
                for cat, count in metrics["grammar_error_categories"].items():
                    formatted_text += f"  {cat}: {count} errors\n"
            formatted_text += "\n"
        
        # Comparison metrics
        if any(m in metrics for m in ["rouge1_f1", "bleu1", "content_similarity"]):
            formatted_text += "COMPARISON WITH REFERENCE:\n" + "-" * 40 + "\n"
            for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bleu1", "bleu2", "bleu3", "bleu4", 
                          "sacrebleu", "content_similarity", "bert_score_f1"]:
                if metric in metrics:
                    value = metrics[metric]
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    formatted_text += f"{metric}: {value_str}"
                    if include_descriptions and metric in descriptions:
                        formatted_text += f" ({descriptions[metric]})"
                    formatted_text += "\n"
            formatted_text += "\n"
        
        # Style comparison metrics
        if any(m in metrics for m in ["sentence_length_similarity", "overall_style_similarity"]):
            formatted_text += "STYLE SIMILARITY TO REFERENCE:\n" + "-" * 40 + "\n"
            for metric in ["sentence_length_similarity", "sentence_distribution_similarity", 
                          "passive_voice_similarity", "adverb_ratio_similarity", 
                          "adjective_ratio_similarity", "transition_phrase_similarity", 
                          "overall_style_similarity"]:
                if metric in metrics:
                    value = metrics[metric]
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    formatted_text += f"{metric}: {value_str}"
                    if include_descriptions and metric in descriptions:
                        formatted_text += f" ({descriptions[metric]})"
                    formatted_text += "\n"
            formatted_text += "\n"
        
        formatted_text += "=" * 80
        return formatted_text
    
    def calculate_perplexity(
        self, 
        text: str, 
        model_name: str = "gpt2"
    ) -> float:
        """
        Calculate perplexity of text using a language model.
        
        Args:
            text: Text to evaluate
            model_name: Pretrained model to use for perplexity calculation
            
        Returns:
            Perplexity score (lower is better)
        """
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # Tokenize text
            encodings = tokenizer(text, return_tensors="pt").to(device)
            
            # Calculate perplexity
            max_length = model.config.max_position_embeddings
            stride = 512
            
            nlls = []
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, encodings.input_ids.size(1))
                trg_len = end_loc - i
                
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
            
            return torch.exp(torch.stack(nlls).sum() / end_loc).item()
        
        except Exception as e:
            print(f"Warning: Error calculating perplexity: {e}")
            return float('nan')


class BlogEvaluator:
    """
    Comprehensive evaluation system for BlogWriter's output.
    
    This class provides functionality to evaluate blog content against references,
    generate detailed reports, and visualize evaluation results.
    """
    
    def __init__(
        self,
        output_dir: str = "./evaluation_results",
        reference_dir: Optional[str] = None,
        style_metrics_path: Optional[str] = None,
        use_grammar_check: bool = True,
        use_bert_score: bool = False
    ):
        """
        Initialize the blog evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            reference_dir: Directory containing reference texts
            style_metrics_path: Path to JSON file with target style metrics
            use_grammar_check: Whether to use grammar checking
            use_bert_score: Whether to use BERTScore
        """
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.reference_texts = []
        self.style_metrics = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize text evaluator
        self.evaluator = TextEvaluator(
            use_grammar_check=use_grammar_check,
            use_bert_score=use_bert_score
        )
        
        # Load reference texts if provided
        if reference_dir and os.path.isdir(reference_dir):
            self._load_reference_texts()
        
        # Load style metrics if provided
        if style_metrics_path and os.path.isfile(style_metrics_path):
            self._load_style_metrics(style_metrics_path)
    
    def _load_reference_texts(self) -> None:
        """Load reference texts from the reference directory."""
        self.reference_texts = []
        for filename in os.listdir(self.reference_dir):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(self.reference_dir, filename), "r", encoding="utf-8") as f:
                        content = f.read()
                        self.reference_texts.append(content)
                except Exception as e:
                    print(f"Warning: Could not load reference file {filename}: {e}")
        
        print(f"Loaded {len(self.reference_texts)} reference texts")
    
    def _load_style_metrics(self, style_metrics_path: str) -> None:
        """Load style metrics from a JSON file."""
        try:
            with open(style_metrics_path, "r", encoding="utf-8") as f:
                self.style_metrics = json.load(f)
            print("Loaded style metrics for comparison")
        except Exception as e:
            print(f"Warning: Could not load style metrics: {e}")
    
    def evaluate(
        self, 
        generated_text: str,
        reference_texts: Optional[List[str]] = None,
        style_metrics: Optional[Dict[str, Any]] = None,
        run_all_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate generated text comprehensively.
        
        Args:
            generated_text: Text to evaluate
            reference_texts: List of reference texts to compare against (overrides instance references)
            style_metrics: Target style metrics (overrides instance metrics)
            run_all_metrics: Whether to run all available metrics (slower)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use provided references/metrics or fall back to instance variables
        refs = reference_texts if reference_texts is not None else self.reference_texts
        style = style_metrics if style_metrics is not None else self.style_metrics
        
        # Perform evaluation
        metrics = self.evaluator.evaluate_text(
            generated_text=generated_text,
            reference_texts=refs,
            style_metrics=style,
            run_all_metrics=run_all_metrics
        )
        
        return metrics
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        output_file: Optional[str] = None,
        include_descriptions: bool = True,
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_file: File to save the report (if None, doesn't save)
            include_descriptions: Whether to include metric descriptions
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report text
        """
        # Format metrics as text
        report_text = TextEvaluator.format_metrics(metrics, include_descriptions)
        
        # Save report if output file is specified
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_text)
                print(f"Evaluation report saved to {output_file}")
                
                # Generate visualizations if requested
                if include_visualizations:
                    visualization_path = output_file.replace(".txt", "") + "_visualizations.png"
                    self._generate_visualizations(metrics, visualization_path)
            except Exception as e:
                print(f"Warning: Could not save report: {e}")
        
        return report_text
    
    def _generate_visualizations(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Generate visualizations of evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_path: Path to save the visualization
        """
        # Set up the visualization
        plt.figure(figsize=(15, 15))
        
        # 1. Style metrics
        if "sentence_categories" in metrics:
            plt.subplot(3, 2, 1)
            cats = metrics["sentence_categories"]
            plt.pie([cats.get("short_sentences", 0), 
                     cats.get("medium_sentences", 0), 
                     cats.get("long_sentences", 0)],
                   labels=["Short", "Medium", "Long"],
                   autopct='%1.1f%%',
                   colors=sns.color_palette("pastel"))
            plt.title("Sentence Length Distribution")
        
        # 2. Readability scores
        readability_metrics = {
            "Flesch": metrics.get("flesch_reading_ease", 0) / 100,  # Normalize to 0-1
            "Dale-Chall": max(0, 1 - metrics.get("dale_chall_readability_score", 0) / 10),  # Inverse and normalize
            "SMOG": max(0, 1 - metrics.get("smog_index", 0) / 20),  # Inverse and normalize
            "ARI": max(0, 1 - metrics.get("automated_readability_index", 0) / 20)  # Inverse and normalize
        }
        
        plt.subplot(3, 2, 2)
        plt.bar(readability_metrics.keys(), readability_metrics.values(), color=sns.color_palette("muted"))
        plt.title("Readability Scores (Higher is Better)")
        plt.ylim(0, 1)
        
        # 3. ROUGE and BLEU scores if available
        comparison_metrics = {
            "ROUGE-1": metrics.get("rouge1_f1", 0),
            "ROUGE-2": metrics.get("rouge2_f1", 0),
            "ROUGE-L": metrics.get("rougeL_f1", 0),
            "BLEU-1": metrics.get("bleu1", 0),
            "BLEU-4": metrics.get("bleu4", 0),
            "Content Similarity": metrics.get("content_similarity", 0)
        }
        
        if any(val > 0 for val in comparison_metrics.values()):
            plt.subplot(3, 2, 3)
            plt.bar(comparison_metrics.keys(), comparison_metrics.values(), color=sns.color_palette("deep"))
            plt.title("Reference Comparison Metrics")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        # 4. Style comparison if available
        style_similarity_metrics = {
            "Sentence Length": metrics.get("sentence_length_similarity", 0),
            "Sentence Distribution": metrics.get("sentence_distribution_similarity", 0),
            "Passive Voice": metrics.get("passive_voice_similarity", 0),
            "Adverb Usage": metrics.get("adverb_ratio_similarity", 0),
            "Adjective Usage": metrics.get("adjective_ratio_similarity", 0),
            "Transitions": metrics.get("transition_phrase_similarity", 0),
            "Overall": metrics.get("overall_style_similarity", 0)
        }
        
        if any(val > 0 for val in style_similarity_metrics.values()):
            plt.subplot(3, 2, 4)
            plt.bar(style_similarity_metrics.keys(), style_similarity_metrics.values(), color=sns.color_palette("bright"))
            plt.title("Style Similarity Metrics")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        # 5. Grammar metrics if available
        if "grammar_error_density" in metrics:
            plt.subplot(3, 2, 5)
            plt.bar(["Grammar Error Density"], [metrics["grammar_error_density"]], color=sns.color_palette("pastel")[2])
            plt.title("Grammar Errors per 100 Words")
            plt.ylim(0, max(5, metrics["grammar_error_density"] * 1.5))  # Dynamic scale with upper limit
        
        # 6. Part of speech distribution
        pos_distribution = {
            "Adverbs": metrics.get("adverb_ratio", 0),
            "Adjectives": metrics.get("adjective_ratio", 0),
            "Passive Voice": metrics.get("passive_voice_ratio", 0)
        }
        
        plt.subplot(3, 2, 6)
        plt.bar(pos_distribution.keys(), pos_distribution.values(), color=sns.color_palette("muted"))
        plt.title("Language Style Indicators")
        plt.ylim(0, max(0.3, max(pos_distribution.values()) * 1.2))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Evaluation visualizations saved to {output_path}")

    def batch_evaluate(
        self,
        generated_texts: Dict[str, str],
        reference_texts: Optional[Dict[str, List[str]]] = None,
        style_metrics: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        include_visualizations: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple generated texts and produce reports.
        
        Args:
            generated_texts: Dictionary of {name: text} pairs to evaluate
            reference_texts: Dictionary of {name: [references]} (optional)
            style_metrics: Target style metrics
            output_dir: Directory to save reports (defaults to instance output_dir)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Dictionary of {name: metrics} for all evaluated texts
        """
        results = {}
        save_dir = output_dir if output_dir else self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Process each text
        for name, text in generated_texts.items():
            # Get references for this text if available
            refs = reference_texts.get(name, self.reference_texts) if reference_texts else self.reference_texts
            
            # Evaluate
            metrics = self.evaluate(text, refs, style_metrics)
            results[name] = metrics
            
            # Generate report
            report_path = os.path.join(save_dir, f"{name}_evaluation.txt")
            self.generate_report(
                metrics, 
                output_file=report_path,
                include_visualizations=include_visualizations
            )
        
        # Generate comparison report if multiple texts
        if len(generated_texts) > 1:
            self._generate_comparison_report(results, save_dir)
        
        return results
    
    def _generate_comparison_report(
        self,
        all_metrics: Dict[str, Dict[str, Any]],
        output_dir: str
    ) -> None:
        """
        Generate a comparison report for multiple evaluated texts.
        
        Args:
            all_metrics: Dictionary of {name: metrics} for all evaluated texts
            output_dir: Directory to save the report
        """
        # Prepare data for comparison
        comparison_data = {}
        
        # Select key metrics for comparison
        key_metrics = [
            "lexical_diversity",
            "flesch_reading_ease",
            "avg_sentence_length",
            "rouge1_f1",
            "bleu4",
            "content_similarity",
            "overall_style_similarity",
            "grammar_error_density"
        ]
        
        # Extract metrics for each text
        for name, metrics in all_metrics.items():
            for metric in key_metrics:
                if metric in metrics:
                    if metric not in comparison_data:
                        comparison_data[metric] = {}
                    comparison_data[metric][name] = metrics[metric]
        
        # Generate comparison visualizations
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(key_metrics):
            if metric in comparison_data and len(comparison_data[metric]) > 0:
                plt.subplot(3, 3, i + 1)
                plt.bar(
                    comparison_data[metric].keys(),
                    comparison_data[metric].values(),
                    color=sns.color_palette("husl", len(comparison_data[metric]))
                )
                plt.title(metric)
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_visualization.png"))
        plt.close()
        
        # Generate CSV report
        comparison_df = pd.DataFrame.from_dict(comparison_data)
        comparison_df.to_csv(os.path.join(output_dir, "comparison_report.csv"))
        
        print(f"Comparison report saved to {output_dir}")


def main():
    """Command-line interface for blog text evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BlogWriter's generated content")
    
    # Input options
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the generated text file to evaluate"
    )
    parser.add_argument(
        "--references", type=str, default=None,
        help="Directory containing reference text files for comparison"
    )
    parser.add_argument(
        "--style_metrics", type=str, default=None,
        help="Path to JSON file with target style metrics"
    )
    
    # Output options
    parser.add_argument(
        "--output", type=str, default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--report_file", type=str, default=None,
        help="Filename for the evaluation report (default: input filename + '_evaluation.txt')"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization plots of evaluation metrics"
    )
    
    # Evaluation options
    parser.add_argument(
        "--grammar_check", action="store_true",
        help="Enable grammar checking (requires language-tool-python)"
    )
    parser.add_argument(
        "--bert_score", action="store_true",
        help="Use BERTScore for semantic similarity (requires bert-score)"
    )
    parser.add_argument(
        "--all_metrics", action="store_true",
        help="Run all available metrics (slower)"
    )
    parser.add_argument(
        "--perplexity", action="store_true",
        help="Calculate perplexity (requires transformers and a language model)"
    )
    parser.add_argument(
        "--perplexity_model", type=str, default="gpt2",
        help="Model to use for perplexity calculation (default: gpt2)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BlogEvaluator(
        output_dir=args.output,
        reference_dir=args.references,
        style_metrics_path=args.style_metrics,
        use_grammar_check=args.grammar_check,
        use_bert_score=args.bert_score
    )
    
    # Read input text
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            generated_text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Determine report filename
    if args.report_file:
        report_path = os.path.join(args.output, args.report_file)
    else:
        input_basename = os.path.basename(args.input)
        report_path = os.path.join(args.output, f"{os.path.splitext(input_basename)[0]}_evaluation.txt")
    
    # Run evaluation
    metrics = evaluator.evaluate(
        generated_text=generated_text,
        run_all_metrics=args.all_metrics
    )
    
    # Calculate perplexity if requested
    if args.perplexity:
        perplexity = evaluator.evaluator.calculate_perplexity(
            text=generated_text,
            model_name=args.perplexity_model
        )
        metrics["perplexity"] = perplexity
    
    # Generate report
    report = evaluator.generate_report(
        metrics=metrics,
        output_file=report_path,
        include_visualizations=args.visualize
    )
    
    # Print summary
    print("\nEVALUATION SUMMARY")
    print("=" * 40)
    print(f"Input text: {args.input}")
    print(f"Word count: {metrics.get('word_count', 0)}")
    print(f"Lexical diversity: {metrics.get('lexical_diversity', 0):.3f}")
    print(f"Readability (Flesch): {metrics.get('flesch_reading_ease', 0):.1f}")
    
    if "overall_style_similarity" in metrics:
        print(f"Style similarity: {metrics.get('overall_style_similarity', 0):.3f}")
    
    if "bleu4" in metrics:
        print(f"BLEU-4 score: {metrics.get('bleu4', 0):.4f}")
    
    if "rouge1_f1" in metrics:
        print(f"ROUGE-1 F1: {metrics.get('rouge1_f1', 0):.4f}")
    
    if "perplexity" in metrics:
        print(f"Perplexity: {metrics.get('perplexity', 0):.2f}")
    
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()