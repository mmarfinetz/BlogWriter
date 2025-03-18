import os
import json
import re
import string
import nltk
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from .style_config import StyleConfig, TRANSITION_PHRASES, PASSIVE_INDICATORS

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class DataProcessor:
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, style_config_path: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the writing samples
            tokenizer: HuggingFace tokenizer for the model
            style_config_path: Path to custom style configuration file
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.style_config = StyleConfig(style_config_path)
        self.style_metrics = None
        
    def load_raw_data(self) -> List[str]:
        """
        Load writing samples from the data directory.
        
        Returns:
            List of text samples
        """
        samples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.data_dir, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    samples.append(content)
                    
        if not samples:
            raise ValueError(f"No text samples found in {self.data_dir}")
                    
        return samples
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into segments of appropriate length.
        
        Args:
            text: Input text
            max_length: Maximum length of each segment in tokens
            
        Returns:
            List of text segments
        """
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
            combined_tokens = len(self.tokenizer.encode(combined))
            
            if combined_tokens > max_length and current_segment:
                segments.append(current_segment)
                current_segment = para
            else:
                current_segment = combined
                
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    def prepare_dataset(self, val_size: float = 0.15) -> DatasetDict:
        """
        Prepare the dataset for training and evaluation.
        
        Args:
            val_size: Proportion of data to use for validation
            
        Returns:
            DatasetDict with train and validation splits
        """
        # Load and clean data
        samples = self.load_raw_data()
        cleaned_samples = [self.clean_text(sample) for sample in samples]
        
        # Segment text
        all_segments = []
        for sample in cleaned_samples:
            segments = self.segment_text(sample)
            all_segments.extend(segments)
            
        # Create train/val splits
        train_segments, val_segments = train_test_split(
            all_segments, test_size=val_size, random_state=42
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_segments})
        val_dataset = Dataset.from_dict({"text": val_segments})
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize examples for model training.
        
        Args:
            examples: Dictionary with text examples
            
        Returns:
            Dictionary with tokenized examples
        """
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )
    
    def prepare_for_training(self) -> DatasetDict:
        """
        Prepare the full dataset for training, including tokenization.
        Analyzes writing style if not already done.
        
        Returns:
            Tokenized dataset ready for training
        """
        # Get raw data samples
        samples = self.load_raw_data()
        cleaned_samples = [self.clean_text(sample) for sample in samples]
        
        # Analyze writing style if not already done
        if self.style_metrics is None:
            print("Analyzing writing style...")
            self.style_metrics = self.analyze_writing_style(cleaned_samples)
        
        dataset = self.prepare_dataset()
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
        
    def analyze_writing_style(self, text_samples: List[str]) -> Dict[str, Any]:
        """
        Analyze writing style metrics from text samples.
        
        Args:
            text_samples: List of cleaned text samples
            
        Returns:
            Dictionary of style metrics
        """
        all_sentences = []
        all_words = []
        sent_lengths = []
        word_lengths = []
        punctuation_freq = defaultdict(int)
        sentence_starters = Counter()
        transition_phrase_counts = Counter()
        passive_voice_count = 0
        total_sentences = 0
        
        # Get style parameters
        params = self.style_config.get_params()
        
        # Process each sample
        for text in text_samples:
            # Split into sentences
            sentences = sent_tokenize(text)
            all_sentences.extend(sentences)
            total_sentences += len(sentences)
            
            # Process each sentence
            for sentence in sentences:
                # Get sentence length
                words = word_tokenize(sentence)
                words_no_punct = [word for word in words if word not in string.punctuation]
                sent_lengths.append(len(words_no_punct))
                all_words.extend(words_no_punct)
                
                # Count word lengths
                for word in words_no_punct:
                    word_lengths.append(len(word))
                
                # Count punctuation
                for char in sentence:
                    if char in string.punctuation:
                        punctuation_freq[char] += 1
                
                # Check for sentence starters (first 1-2 words)
                if words_no_punct:
                    starter = words_no_punct[0].lower()
                    if len(words_no_punct) > 1:
                        starter_bigram = f"{words_no_punct[0].lower()} {words_no_punct[1].lower()}"
                        sentence_starters[starter_bigram] += 1
                    sentence_starters[starter] += 1
                
                # Check for transition phrases
                lower_sent = sentence.lower()
                for phrase in TRANSITION_PHRASES:
                    if phrase in lower_sent:
                        transition_phrase_counts[phrase] += 1
                
                # Check for passive voice indicators
                for indicator in PASSIVE_INDICATORS:
                    if indicator in f" {lower_sent} ":
                        # This is a simple heuristic - more accurate would use POS tagging
                        passive_voice_count += 1
                        break
        
        # Calculate overall lexical diversity
        word_count = len(all_words)
        unique_words = len(set(w.lower() for w in all_words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Tag words with part of speech
        tagged_words = nltk.pos_tag(all_words)
        pos_counts = Counter(tag for word, tag in tagged_words)
        
        # Count adverbs (RB tags)
        adverb_count = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
        
        # Calculate vocabulary complexity metrics
        simple_word_threshold = params['vocabulary']['simple_word_length']
        complex_word_threshold = params['vocabulary']['complex_word_length']
        
        word_complexity = {
            'simple_words': sum(1 for l in word_lengths if l <= simple_word_threshold),
            'medium_words': sum(1 for l in word_lengths if simple_word_threshold < l < complex_word_threshold),
            'complex_words': sum(1 for l in word_lengths if l >= complex_word_threshold)
        }
        
        # Calculate sentence length distribution
        short_threshold = params['sentence_length']['short_sentence']
        long_threshold = params['sentence_length']['long_sentence']
        
        sentence_distribution = {
            'short_sentences': sum(1 for l in sent_lengths if l <= short_threshold),
            'medium_sentences': sum(1 for l in sent_lengths if short_threshold < l < long_threshold),
            'long_sentences': sum(1 for l in sent_lengths if l >= long_threshold)
        }
        
        # Compile metrics
        metrics = {
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': np.mean(sent_lengths) if sent_lengths else 0,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'sentence_length_std': np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
            'word_length_distribution': Counter(word_lengths),
            'sentence_length_distribution': Counter(sent_lengths),
            'sentence_categories': sentence_distribution,
            'word_complexity': word_complexity,
            'punctuation_frequency': dict(punctuation_freq),
            'top_sentence_starters': dict(sentence_starters.most_common(20)),
            'top_transition_phrases': dict(transition_phrase_counts.most_common(20)),
            'passive_voice_ratio': passive_voice_count / total_sentences if total_sentences > 0 else 0,
            'adverb_ratio': adverb_count / word_count if word_count > 0 else 0,
            'pos_distribution': {tag: count/word_count for tag, count in pos_counts.items()} if word_count > 0 else {},
            'total_words': word_count,
            'unique_words': unique_words,
            'total_sentences': total_sentences
        }
        
        return metrics
    
    def save_style_metrics(self, output_dir: str) -> str:
        """
        Save style metrics to a JSON file.
        
        Args:
            output_dir: Directory to save the style metrics
            
        Returns:
            Path to the saved metrics file
        """
        if self.style_metrics is None:
            raise ValueError("Style metrics have not been calculated yet.")
        
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "style_metrics.json")
        
        # Convert numpy values to Python native types for JSON serialization
        serializable_metrics = {}
        
        for key, value in self.style_metrics.items():
            if isinstance(value, dict):
                serializable_metrics[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
            elif isinstance(value, np.number):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2)
            
        print(f"Style metrics saved to {metrics_path}")
        return metrics_path
    
    def visualize_style(self, output_dir: str = None) -> None:
        """
        Visualize writing style characteristics.
        
        Args:
            output_dir: Optional directory to save visualization files
        """
        if self.style_metrics is None:
            raise ValueError("Style metrics have not been calculated yet.")
        
        metrics = self.style_metrics
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Sentence Length Distribution
        ax1 = fig.add_subplot(2, 2, 1)
        sent_lengths = sorted(metrics['sentence_length_distribution'].items())
        ax1.bar([str(x) for x, _ in sent_lengths], [y for _, y in sent_lengths])
        ax1.set_title('Sentence Length Distribution')
        ax1.set_xlabel('Words per Sentence')
        ax1.set_ylabel('Frequency')
        
        # Simplify x-axis if too many bars
        if len(sent_lengths) > 15:
            ax1.set_xticks(range(0, len(sent_lengths), len(sent_lengths) // 10))
        
        # 2. Word Complexity
        ax2 = fig.add_subplot(2, 2, 2)
        complexity_labels = ['Simple', 'Medium', 'Complex']
        complexity_values = [
            metrics['word_complexity']['simple_words'],
            metrics['word_complexity']['medium_words'],
            metrics['word_complexity']['complex_words']
        ]
        ax2.pie(complexity_values, labels=complexity_labels, autopct='%1.1f%%')
        ax2.set_title('Word Complexity Distribution')
        
        # 3. Top Punctuation
        ax3 = fig.add_subplot(2, 2, 3)
        punct_items = sorted(
            metrics['punctuation_frequency'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:8]
        ax3.bar([x for x, _ in punct_items], [y for _, y in punct_items])
        ax3.set_title('Punctuation Usage')
        ax3.set_xlabel('Punctuation Mark')
        ax3.set_ylabel('Frequency')
        
        # 4. Sentence Structure
        ax4 = fig.add_subplot(2, 2, 4)
        struct_labels = ['Short', 'Medium', 'Long']
        struct_values = [
            metrics['sentence_categories']['short_sentences'],
            metrics['sentence_categories']['medium_sentences'],
            metrics['sentence_categories']['long_sentences']
        ]
        ax4.pie(struct_values, labels=struct_labels, autopct='%1.1f%%')
        ax4.set_title('Sentence Length Categories')
        
        plt.tight_layout()
        
        # Save or display the figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, 'style_visualization.png')
            plt.savefig(fig_path)
            print(f"Style visualization saved to {fig_path}")
        else:
            plt.show()
        
        plt.close(fig)