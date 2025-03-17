import os
from typing import Dict, List, Tuple
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

class DataProcessor:
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the writing samples
            tokenizer: HuggingFace tokenizer for the model
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        
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
        
        Returns:
            Tokenized dataset ready for training
        """
        dataset = self.prepare_dataset()
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset