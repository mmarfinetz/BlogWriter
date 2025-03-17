from typing import List, Dict, Optional, Any, Union
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class BlogGenerator:
    def __init__(self, model_path: str):
        """
        Initialize the blog post generator.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.model_path = model_path
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate(
        self,
        prompt: str,
        max_length: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate blog content based on a prompt.
        
        Args:
            prompt: Starting text for generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative, lower = more focused)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            num_return_sequences: Number of texts to generate
            
        Returns:
            List of generated texts
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode and return generated texts
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return generated_texts
    
    def save_generated_text(self, text: str, filename: str) -> None:
        """
        Save generated text to a file.
        
        Args:
            text: Generated text
            filename: Name of the file to save the text
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"Generated text saved to {filename}")