from typing import List, Dict, Optional, Any, Union
import os
import json
import re
import torch
import nltk
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
from .style_config import StyleConfig, TRANSITION_PHRASES
from .templates import TemplateManager, ContentType, StyleType, ToneType, FormatType

class BlogGenerator:
    def __init__(self, model_path: str, style_aware: bool = True, templates_dir: str = None, data_dir: str = None):
        """
        Initialize the blog post generator.
        
        Args:
            model_path: Path to the fine-tuned model or LoRA adapter
            style_aware: Whether to use style metrics for generation guidance
            templates_dir: Directory containing custom templates
            data_dir: Directory containing example content for few-shot learning
        """
        self.model_path = model_path
        self.style_aware = style_aware
        self.style_metrics = None
        self.data_dir = data_dir
        self.example_texts = []
        
        # Check if this is a LoRA model by looking for lora_config.json
        lora_config_path = os.path.join(model_path, "lora_config.json")
        is_lora_model = os.path.exists(lora_config_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if is_lora_model:
            # Load LoRA configuration
            with open(lora_config_path, "r") as f:
                lora_config = json.load(f)
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(lora_config["base_model_name"])
            
            # Then load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print(f"Loaded LoRA model with rank {lora_config['r']}, alpha {lora_config['alpha']}")
        else:
            # Standard model loading
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Load style metrics if they exist and style awareness is enabled
        if style_aware:
            style_metrics_path = os.path.join(model_path, "style_metrics.json")
            if os.path.exists(style_metrics_path):
                try:
                    with open(style_metrics_path, "r", encoding="utf-8") as f:
                        self.style_metrics = json.load(f)
                    print("Loaded writing style metrics for style-aware generation")
                except Exception as e:
                    print(f"Warning: Could not load style metrics: {e}")
            else:
                print("Warning: No style metrics found. Generation will not be style-aware.")
                
        # Load style configuration
        self.style_config = StyleConfig()
        
        # Initialize template manager
        self.template_manager = TemplateManager(templates_dir)
        
        # Load example texts if data directory is provided
        if data_dir and os.path.isdir(data_dir):
            self._load_example_texts()
        
    def generate(
        self,
        prompt: str,
        max_length: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        style_strength: float = 0.8,
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
            style_strength: How strongly to enforce style (0-1, higher means stronger enforcement)
            
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
        
        # Decode generated texts
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Apply style-aware post-processing if enabled and metrics are available
        if self.style_aware and self.style_metrics is not None and style_strength > 0:
            processed_texts = []
            for text in generated_texts:
                processed_text = self.apply_style_guidance(text, style_strength)
                processed_texts.append(processed_text)
            return processed_texts
        
        return generated_texts
    
    def analyze_text_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze the style characteristics of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of style metrics for the text
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Calculate sentence lengths
        sent_lengths = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words_no_punct = [word for word in words if word not in '.,!?;:()[]{}""\'']
            sent_lengths.append(len(words_no_punct))
        
        # Calculate average sentence length
        avg_sentence_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        # Count transition phrases
        transition_count = 0
        for phrase in TRANSITION_PHRASES:
            if phrase in text.lower():
                transition_count += 1
        
        # Count punctuation
        punct_count = {}
        for char in text:
            if char in '.,!?;:()[]{}""\'':
                punct_count[char] = punct_count.get(char, 0) + 1
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'sentence_lengths': sent_lengths,
            'transition_phrases': transition_count,
            'punctuation': punct_count
        }
    
    def apply_style_guidance(self, text: str, strength: float = 0.8) -> str:
        """
        Apply style guidance to generated text based on learned style metrics.
        
        Args:
            text: Generated text to process
            strength: How strongly to apply style guidance (0-1)
            
        Returns:
            Style-adjusted text
        """
        if not self.style_metrics:
            return text
        
        # Start with basic adjustments that don't require complex transformations
        
        # 1. Split into sections for processing
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 3:
            return text  # Too short to meaningfully adjust
        
        # 2. Get current text style metrics
        current_style = self.analyze_text_style(text)
        target_avg_sentence_length = self.style_metrics.get('avg_sentence_length', 0)
        
        # 3. Apply sentence length adjustment if needed
        if abs(current_style['avg_sentence_length'] - target_avg_sentence_length) > 5:
            # We need to adjust sentence lengths
            adjusted_sentences = []
            
            for i, sentence in enumerate(sentences):
                if len(adjusted_sentences) > 0 and i < len(sentences) - 1:
                    # Check if we should combine with next or previous for longer sentences
                    if target_avg_sentence_length > current_style['avg_sentence_length'] + 3:
                        # Need longer sentences - try to combine when appropriate
                        words = nltk.word_tokenize(sentence)
                        next_words = nltk.word_tokenize(sentences[i+1])
                        
                        # Only combine if it makes logical sense (e.g., with conjunctions)
                        if len(words) < target_avg_sentence_length and (
                            next_words[0].lower() in ['and', 'but', 'or', 'so', 'because', 'however'] or
                            words[-1] in [',', ':'] or 
                            len(words) + len(next_words) < target_avg_sentence_length * 1.5
                        ):
                            # Replace period with comma or semicolon
                            if sentence.endswith('.'):
                                new_sent = sentence[:-1] + '; ' + sentences[i+1]
                            else:
                                new_sent = sentence + ' ' + sentences[i+1]
                            
                            adjusted_sentences.append(new_sent)
                            # Skip the next sentence as we've combined it
                            sentences[i+1] = ""
                            continue
                
                # If we didn't combine, keep original
                if sentence:
                    adjusted_sentences.append(sentence)
            
            # Rebuild text from adjusted sentences
            text = ' '.join(adjusted_sentences)
        
        # 4. Add transition phrases if needed according to user's style
        if self.style_metrics.get('top_transition_phrases') and strength > 0.5:
            # Get favorite transition types from the style metrics
            favorite_transitions = list(self.style_metrics.get('top_transition_phrases', {}).keys())
            
            if favorite_transitions:
                sentences = nltk.sent_tokenize(text)
                enhanced_sentences = []
                
                # Only add transitions to some sentences based on strength
                modify_count = int(len(sentences) * 0.2 * strength)
                modify_indices = [i for i in range(1, len(sentences)) 
                                 if i % max(2, int(len(sentences)/modify_count)) == 0]
                
                import random
                for i, sentence in enumerate(sentences):
                    if i in modify_indices and i > 0:
                        # Add a transition at the beginning of this sentence
                        transition = random.choice(favorite_transitions[:3])
                        sentence = f"{transition.capitalize()}, {sentence[0].lower()}{sentence[1:]}"
                    
                    enhanced_sentences.append(sentence)
                
                text = ' '.join(enhanced_sentences)
        
        return text
        
    def get_style_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the detected writing style characteristics.
        
        Returns:
            Dictionary with key style indicators
        """
        if not self.style_metrics:
            return {"error": "No style metrics available"}
        
        return {
            "avg_sentence_length": self.style_metrics.get("avg_sentence_length", 0),
            "lexical_diversity": self.style_metrics.get("lexical_diversity", 0),
            "sentence_distribution": {
                "short": self.style_metrics.get("sentence_categories", {}).get("short_sentences", 0),
                "medium": self.style_metrics.get("sentence_categories", {}).get("medium_sentences", 0),
                "long": self.style_metrics.get("sentence_categories", {}).get("long_sentences", 0)
            },
            "top_transitions": list(self.style_metrics.get("top_transition_phrases", {}).keys())[:5],
            "passive_voice_ratio": self.style_metrics.get("passive_voice_ratio", 0),
            "adverb_ratio": self.style_metrics.get("adverb_ratio", 0)
        }
    
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
    
    def _load_example_texts(self) -> None:
        """
        Load example texts from the data directory for few-shot learning.
        """
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return
            
        self.example_texts = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(self.data_dir, filename), "r", encoding="utf-8") as f:
                        content = f.read()
                        self.example_texts.append(content)
                except Exception as e:
                    print(f"Warning: Could not load example file {filename}: {e}")
                    
        print(f"Loaded {len(self.example_texts)} example texts for few-shot learning")
    
    def _find_relevant_examples(self, topic: str, content_type: str, num_examples: int = 2) -> List[str]:
        """
        Find the most relevant examples based on topic and content type.
        
        Args:
            topic: The main topic
            content_type: Type of content to generate
            num_examples: Number of examples to return
            
        Returns:
            List of relevant example texts
        """
        if not self.example_texts:
            return []
            
        # Simple content-based filtering
        # In a real implementation, this would use more sophisticated methods
        # such as embedding similarity or semantic search
        
        # Create a set of keywords from the topic
        topic_words = set(re.findall(r'\w+', topic.lower()))
        
        # Score examples based on keyword matches
        scored_examples = []
        for example in self.example_texts:
            # Count word matches
            example_words = set(re.findall(r'\w+', example.lower()))
            match_score = len(topic_words.intersection(example_words))
            
            # Higher score for examples that match the content type
            if content_type.lower() in example.lower():
                match_score += 5
                
            scored_examples.append((match_score, example))
        
        # Sort by score and return top N
        scored_examples.sort(reverse=True)
        return [example for _, example in scored_examples[:num_examples]]
        
    def generate_with_template(
        self,
        topic: str,
        content_type: str = "blog",
        style: str = None,
        tone: str = None,
        format_type: str = None,
        max_length: int = 1200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        style_strength: float = 0.8,
        use_examples: bool = True,
        num_examples: int = 2,
        additional_instructions: str = "",
        maintain_consistency: bool = True
    ) -> List[str]:
        """
        Generate content using templates and style parameters.
        
        Args:
            topic: Main topic for generation
            content_type: Type of content (blog, essay, technical, newsletter)
            style: Writing style (formal, casual, academic, conversational, professional)
            tone: Content tone (informative, persuasive, entertaining, authoritative, friendly, analytical)
            format_type: Content format (standard, listicle, guide, review, opinion, comparison)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative, lower = more focused)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            num_return_sequences: Number of texts to generate
            style_strength: How strongly to enforce style (0-1, higher means stronger enforcement)
            use_examples: Whether to use few-shot examples
            num_examples: Number of examples to use if few-shot learning is enabled
            additional_instructions: Additional specific instructions for generation
            maintain_consistency: Whether to enforce consistent style across the content
            
        Returns:
            List of generated texts
        """
        # 1. Get the appropriate template
        template = self.template_manager.get_template(
            content_type=content_type,
            style=style,
            tone=tone,
            format_type=format_type,
            additional_instructions=additional_instructions
        )
        
        # 2. Find relevant examples if few-shot learning is enabled
        example_texts = []
        if use_examples and self.example_texts:
            example_texts = self._find_relevant_examples(topic, content_type, num_examples)
        
        # 3. Format the template with topic and examples
        formatted_prompt = self.template_manager.format_with_examples(
            template=template,
            topic=topic,
            examples=example_texts,
            num_examples=num_examples
        )
        
        # 4. Generate text using the formatted prompt
        generated_texts = self.generate(
            prompt=formatted_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            style_strength=style_strength
        )
        
        # 5. Apply consistency enforcement if requested
        if maintain_consistency and self.style_aware and len(generated_texts) > 0:
            processed_texts = []
            for text in generated_texts:
                consistent_text = self._apply_consistency_enforcement(text)
                processed_texts.append(consistent_text)
            return processed_texts
        
        return generated_texts
    
    def _apply_consistency_enforcement(self, text: str) -> str:
        """
        Apply techniques to maintain consistent style throughout longer content.
        
        Args:
            text: Generated text to process
            
        Returns:
            Consistency-enhanced text
        """
        # Skip processing if text is too short
        if len(text) < 200:
            return text
            
        # 1. Split text into paragraphs for analysis
        paragraphs = text.split('\n\n')
        if len(paragraphs) <= 2:
            # Not enough paragraphs to meaningfully process
            return text
            
        # 2. Analyze first few paragraphs to establish baseline style
        intro_text = '\n\n'.join(paragraphs[:2])
        intro_style = self.analyze_text_style(intro_text)
        
        # Extract style metrics we want to maintain
        target_sentence_length = intro_style.get('avg_sentence_length', 0)
        
        # 3. Process each paragraph to maintain consistent style
        processed_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            # Skip processing for very short paragraphs
            if len(paragraph.split()) < 20:
                processed_paragraphs.append(paragraph)
                continue
                
            # Analyze current paragraph
            current_style = self.analyze_text_style(paragraph)
            
            # Apply style consistency improvements
            processed_paragraph = paragraph
            
            # 3.1 Adjust sentence length if significantly different
            if abs(current_style.get('avg_sentence_length', 0) - target_sentence_length) > 8:
                sentences = nltk.sent_tokenize(paragraph)
                if len(sentences) > 1:
                    # Apply sentence length adjustments similar to apply_style_guidance
                    # But more targeted to match the intro style
                    adjusted_sentences = []
                    
                    for j, sentence in enumerate(sentences):
                        # Apply adjustments to make sentence length closer to target
                        words = nltk.word_tokenize(sentence)
                        words_no_punct = [word for word in words if word not in '.,!?;:()[]{}""\'']
                        
                        if len(words_no_punct) > target_sentence_length * 1.5 and j < len(sentences) - 1:
                            # Split overly long sentences
                            mid_point = len(words) // 2
                            # Find a good split point near the middle
                            for k in range(mid_point - 3, mid_point + 3):
                                if k < len(words) and words[k] in [',', ';', ':', 'and', 'but', 'or']:
                                    mid_point = k
                                    break
                            
                            first_half = ' '.join(words[:mid_point]) + '.'
                            second_half = ' '.join([word.capitalize() if i == 0 else word 
                                                   for i, word in enumerate(words[mid_point+1:])])
                            
                            adjusted_sentences.append(first_half)
                            adjusted_sentences.append(second_half)
                        
                        elif len(words_no_punct) < target_sentence_length * 0.5 and j < len(sentences) - 1:
                            # Combine short sentences when appropriate
                            next_words = nltk.word_tokenize(sentences[j+1])
                            
                            if next_words[0].lower() in ['and', 'but', 'or', 'so', 'because', 'however'] or \
                               words[-1] in [',', ':']:
                                if sentence.endswith('.'):
                                    new_sent = sentence[:-1] + '; ' + sentences[j+1]
                                else:
                                    new_sent = sentence + ' ' + sentences[j+1]
                                
                                adjusted_sentences.append(new_sent)
                                sentences[j+1] = ""
                            else:
                                adjusted_sentences.append(sentence)
                        
                        else:
                            if sentence:  # Skip empty sentences
                                adjusted_sentences.append(sentence)
                    
                    processed_paragraph = ' '.join(adjusted_sentences)
            
            # 3.2 Add transition phrases between paragraphs for better flow
            if i > 0 and i < len(paragraphs) - 1 and 'top_transition_phrases' in self.style_metrics:
                # Get favorite transition types from the style metrics
                favorite_transitions = list(self.style_metrics.get('top_transition_phrases', {}).keys())
                
                if favorite_transitions and not any(phrase in processed_paragraph.lower() 
                                                 for phrase in favorite_transitions[:5]):
                    # Add a transition at the beginning if none exists
                    first_sentence = nltk.sent_tokenize(processed_paragraph)[0]
                    rest_of_paragraph = processed_paragraph[len(first_sentence):]
                    
                    transition = random.choice(favorite_transitions[:3])
                    modified_first = f"{transition.capitalize()}, {first_sentence[0].lower()}{first_sentence[1:]}"
                    
                    processed_paragraph = modified_first + rest_of_paragraph
            
            processed_paragraphs.append(processed_paragraph)
        
        # 4. Combine processed paragraphs back into text
        return '\n\n'.join(processed_paragraphs)