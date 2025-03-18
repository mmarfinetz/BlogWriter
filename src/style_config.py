from typing import Dict, List, Tuple, Any
import json
import os

# Default parameters for style analysis
DEFAULT_STYLE_PARAMS = {
    # Sentence length parameters
    "sentence_length": {
        "min_length": 5,         # Minimum words in a sentence to count in analysis
        "max_length": 50,        # Maximum words in a sentence to count in analysis
        "short_sentence": 10,    # Threshold for defining short sentences
        "long_sentence": 25      # Threshold for defining long sentences
    },
    
    # Vocabulary complexity parameters
    "vocabulary": {
        "simple_word_length": 5,  # Words with this many chars or fewer considered simple
        "complex_word_length": 8, # Words with this many chars or more considered complex
        "rare_word_threshold": 0.01  # Frequency threshold for rare words
    },
    
    # Style markers
    "style_markers": {
        "track_adverbs": True,              # Whether to track adverb usage
        "track_passive_voice": True,        # Whether to track passive voice usage
        "track_transition_phrases": True,   # Whether to track transition phrases
        "track_sentence_starters": True,    # Whether to track sentence starting patterns
    },
    
    # Generation constraints
    "generation_constraints": {
        "enforce_sentence_length": True,    # Whether to enforce sentence length distributions
        "enforce_vocabulary_mix": True,     # Whether to enforce vocabulary complexity
        "enforce_punctuation": True,        # Whether to enforce punctuation style
        "allow_style_variation": 0.2        # How much variation to allow (0-1)
    }
}

# Common transition phrases to track
TRANSITION_PHRASES = [
    # Contrast transitions
    "however", "nevertheless", "on the other hand", "in contrast", "conversely",
    # Similarity transitions
    "similarly", "likewise", "in the same way", "equally", "just as",
    # Cause and effect
    "therefore", "consequently", "as a result", "thus", "hence",
    # Additional information
    "furthermore", "moreover", "in addition", "additionally", "also",
    # Examples
    "for instance", "for example", "specifically", "to illustrate", "namely",
    # Clarification
    "in other words", "to clarify", "that is", "to put it another way",
    # Conclusion
    "in conclusion", "to summarize", "ultimately", "finally", "in summary"
]

# Common passive voice indicators
PASSIVE_INDICATORS = [
    " is ", " are ", " was ", " were ", " be ", " been ", " being "
]

class StyleConfig:
    """
    Manages writing style configuration and profiles.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize style configuration.
        
        Args:
            config_path: Path to custom style configuration file
        """
        self.params = DEFAULT_STYLE_PARAMS.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_params = json.load(f)
                # Update default params with custom ones
                for category, settings in custom_params.items():
                    if category in self.params:
                        self.params[category].update(settings)
                    else:
                        self.params[category] = settings
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current style parameters.
        
        Returns:
            Dictionary of style parameters
        """
        return self.params
    
    def save_config(self, path: str) -> None:
        """
        Save current configuration to a file.
        
        Args:
            path: Path to save the configuration
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2)
    
    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update style parameters.
        
        Args:
            new_params: New parameter values to set
        """
        for category, settings in new_params.items():
            if category in self.params:
                self.params[category].update(settings)
            else:
                self.params[category] = settings