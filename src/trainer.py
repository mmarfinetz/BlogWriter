from typing import Dict, Optional, Union
import os
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

class BlogModelTrainer:
    def __init__(
        self,
        model_name: str = "gpt2",
        data_dir: str = "./data",
        output_dir: str = "./models",
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name of the pretrained model to use
            data_dir: Directory containing writing samples
            output_dir: Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize data processor
        self.data_processor = DataProcessor(data_dir, self.tokenizer)
        
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
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer
        self.model.save_pretrained(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
        print(f"Model saved to {os.path.join(self.output_dir, 'final_model')}")