from typing import Dict, Optional, Union, Any, List
import os
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EvalPrediction
)
from datasets import DatasetDict
import torch
from .data_processor import DataProcessor
from .style_config import StyleConfig
from .evaluation import TextEvaluator
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

class BlogModelTrainer:
    def __init__(
        self,
        model_name: str = "gpt2",
        data_dir: str = "./data",
        output_dir: str = "./models",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        style_config_path: str = None,
        analyze_style: bool = True,
        evaluate_outputs: bool = True,
        evaluation_dir: str = "./evaluation_results"
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name of the pretrained model to use
            data_dir: Directory containing writing samples
            output_dir: Directory to save the fine-tuned model
            use_lora: Whether to use LoRA for fine-tuning
            lora_r: Rank of the LoRA update matrices
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            style_config_path: Path to custom style configuration file
            analyze_style: Whether to analyze writing style metrics
            evaluate_outputs: Whether to evaluate generated outputs during training
            evaluation_dir: Directory to save evaluation results
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.style_config_path = style_config_path
        self.analyze_style = analyze_style
        self.evaluate_outputs = evaluate_outputs
        self.evaluation_dir = evaluation_dir
        
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Apply LoRA if specified
        if use_lora:
            self.prepare_for_lora()
            
        # Initialize data processor with style config
        self.data_processor = DataProcessor(data_dir, self.tokenizer, style_config_path)
        
        # Initialize text evaluator if evaluation is enabled
        if evaluate_outputs:
            self.text_evaluator = TextEvaluator(use_grammar_check=False, use_bert_score=False)
        
    def prepare_for_lora(self) -> None:
        """
        Prepare the model for LoRA fine-tuning.
        
        This method configures the model with LoRA adapters for efficient
        fine-tuning with a reduced parameter count.
        """
        # Define LoRA Configuration
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"],  # Typical targets for GPT-2 architecture
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters info
        self.model.print_trainable_parameters()
        
    def prepare_data(self) -> DatasetDict:
        """
        Prepare the dataset for training.
        
        Returns:
            Tokenized dataset ready for training
        """
        return self.data_processor.prepare_for_training()
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics for the model.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Metrics dictionary
        """
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate perplexity
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        
        return {"perplexity": perplexity}
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        eval_steps: int = 1000,
        gradient_accumulation_steps: int = 8,
        generate_samples: bool = True,
        num_evaluation_samples: int = 3,
        sample_length: int = 500
    ) -> None:
        """
        Fine-tune the model on the provided data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Steps between model checkpoints
            logging_steps: Steps between logging updates
            eval_steps: Steps between evaluations
            gradient_accumulation_steps: Steps for gradient accumulation
            generate_samples: Whether to generate and evaluate text samples during training
            num_evaluation_samples: Number of samples to generate for evaluation
            sample_length: Maximum length of generated evaluation samples
        """
        # Prepare data
        tokenized_dataset = self.prepare_data()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're using standard language modeling (not masked)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_total_limit=3,  # Keep only the 3 most recent checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="perplexity",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
            report_to="tensorboard",
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics,
        )
        
        # Hook for generating and evaluating samples during training
        if generate_samples and self.evaluate_outputs:
            original_save_model = trainer.save_model
            
            def save_model_with_evaluation(*args, **kwargs):
                # Call the original save_model method
                result = original_save_model(*args, **kwargs)
                
                # Generate samples for evaluation
                self._generate_and_evaluate_samples(
                    num_samples=num_evaluation_samples,
                    max_length=sample_length
                )
                
                return result
            
            # Replace the save_model method with our custom version
            trainer.save_model = save_model_with_evaluation
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer
        final_model_path = os.path.join(self.output_dir, "final_model")
        
        if self.use_lora:
            # For LoRA models, save the adapters separately
            self.model.save_pretrained(final_model_path)
            
            # Save a JSON file indicating this is a LoRA model
            lora_config_path = os.path.join(final_model_path, "lora_config.json")
            with open(lora_config_path, "w") as f:
                json.dump({
                    "base_model_name": self.model_name,
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout
                }, f)
        else:
            # For full models, save everything
            self.model.save_pretrained(final_model_path)
            
        # Save tokenizer in all cases
        self.tokenizer.save_pretrained(final_model_path)
        
        # Save style metrics if they were analyzed
        if self.analyze_style and self.data_processor.style_metrics is not None:
            # Save metrics JSON
            metrics_path = self.data_processor.save_style_metrics(final_model_path)
            
            # Generate and save style visualization
            self.data_processor.visualize_style(final_model_path)
            
            print(f"Writing style metrics saved to {metrics_path}")
            print(f"Style visualization saved to {os.path.join(final_model_path, 'style_visualization.png')}")
        
        # Final evaluation with the fully trained model
        if self.evaluate_outputs and generate_samples:
            print("Performing final evaluation of generated samples...")
            self._generate_and_evaluate_samples(
                num_samples=num_evaluation_samples,
                max_length=sample_length,
                save_prefix="final"
            )
        
        print(f"Model saved to {final_model_path}")
        if self.use_lora:
            print(f"This is a LoRA model with rank {self.lora_r}, alpha {self.lora_alpha}")
            
    def _generate_and_evaluate_samples(
        self,
        num_samples: int = 3,
        max_length: int = 500,
        temperature: float = 0.8,
        prompts: Optional[List[str]] = None,
        save_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Generate and evaluate text samples using the current model state.
        
        Args:
            num_samples: Number of samples to generate
            max_length: Maximum length of each sample
            temperature: Sampling temperature
            prompts: Optional list of prompts to use (if None, generates defaults)
            save_prefix: Prefix for saved files (e.g., checkpoint name)
            
        Returns:
            Dictionary with evaluation metrics for the samples
        """
        if not self.evaluate_outputs:
            return {}
            
        # Default prompts if none provided
        if prompts is None:
            prompts = [
                "The key to effective writing is",
                "A comprehensive guide to understanding",
                "The future of technology looks like"
            ][:num_samples]
        
        # Ensure we have enough prompts
        if len(prompts) < num_samples:
            prompts.extend(["Write a blog post about an interesting topic"] * (num_samples - len(prompts)))
        
        # Get reference texts from training data for comparison
        reference_texts = self._get_reference_texts()
        
        # Generate samples
        samples = {}
        metrics_by_sample = {}
        combined_metrics = {}
        
        for i, prompt in enumerate(prompts[:num_samples]):
            # Generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode generated text
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            sample_name = f"sample_{i+1}"
            samples[sample_name] = generated_text
            
            # Evaluate the sample
            metrics = self.text_evaluator.evaluate_text(
                generated_text=generated_text,
                reference_texts=reference_texts,
                style_metrics=self.data_processor.style_metrics
            )
            
            metrics_by_sample[sample_name] = metrics
            
            # Save sample and its evaluation
            timestamp = save_prefix if save_prefix else "checkpoint"
            sample_file = os.path.join(self.evaluation_dir, f"{timestamp}_{sample_name}.txt")
            report_file = os.path.join(self.evaluation_dir, f"{timestamp}_{sample_name}_evaluation.txt")
            
            # Save generated text
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write(generated_text)
                
            # Save evaluation report
            report = self.text_evaluator.format_metrics(metrics)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
        
        # Calculate aggregated metrics
        for metric in ["lexical_diversity", "flesch_reading_ease", "rouge1_f1", "bleu4", 
                      "content_similarity", "overall_style_similarity"]:
            values = [metrics.get(metric, 0) for metrics in metrics_by_sample.values()]
            if values:
                combined_metrics[f"avg_{metric}"] = sum(values) / len(values)
        
        # Save aggregated metrics
        if combined_metrics:
            agg_file = os.path.join(self.evaluation_dir, f"{save_prefix}_aggregated_metrics.json")
            with open(agg_file, "w", encoding="utf-8") as f:
                json.dump(combined_metrics, f, indent=2)
        
        return combined_metrics
    
    def _get_reference_texts(self, max_texts: int = 5) -> List[str]:
        """
        Get a sample of reference texts from the training data.
        
        Args:
            max_texts: Maximum number of reference texts to retrieve
            
        Returns:
            List of reference text samples
        """
        reference_texts = []
        
        try:
            # Get raw training data from processor
            raw_texts = []
            
            # If data processor has stored samples, use them
            if hasattr(self.data_processor, "samples") and self.data_processor.samples:
                raw_texts = self.data_processor.samples[:max_texts]
            else:
                # Otherwise, look for text files in the data directory
                file_count = 0
                for filename in os.listdir(self.data_dir):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(self.data_dir, filename)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                raw_texts.append(f.read())
                                file_count += 1
                                if file_count >= max_texts:
                                    break
                        except Exception as e:
                            print(f"Warning: Could not read {filename}: {e}")
            
            # Process texts - ensure they're not too long
            for text in raw_texts:
                # Truncate if needed
                max_chars = 2000
                if len(text) > max_chars:
                    # Try to find a sentence boundary near max_chars
                    sentences = text.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated) + len(sentence) + 2 <= max_chars:
                            truncated += sentence + ". "
                        else:
                            break
                    reference_texts.append(truncated)
                else:
                    reference_texts.append(text)
        
        except Exception as e:
            print(f"Warning: Error getting reference texts: {e}")
        
        return reference_texts