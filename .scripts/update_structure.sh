#!/bin/bash
# Simple script to update the project structure documentation

# Create the output file with header
echo "# Project Structure" > .cursor/rules/structure.mdc
echo "" >> .cursor/rules/structure.mdc
echo "This document provides an overview of the BlogWriter project structure." >> .cursor/rules/structure.mdc
echo "" >> .cursor/rules/structure.mdc
echo "\`\`\`" >> .cursor/rules/structure.mdc

# Check if tree command is available
if command -v tree &> /dev/null; then
  # Use tree command for better visualization
  # Skip .git directory and include hidden files/directories except .git
  tree -a -I ".git" --charset=ascii >> .cursor/rules/structure.mdc
  echo "Using tree command for structure visualization."
else
  # Fallback to find if tree is not available
  echo "Tree command not found. Using find command instead."
  find . -type f -not -path "./.git/*" | sort >> .cursor/rules/structure.mdc
fi

# Close the code block
echo "\`\`\`" >> .cursor/rules/structure.mdc

# Add static content about key files
cat >> .cursor/rules/structure.mdc << 'EOF'

## Key Files and Directories

- **train.py**: Main entry point for training models on your writing samples.
- **generate.py**: Main entry point for generating new content in your style.
- **data/**: Place writing samples here for fine-tuning.
- **models/**: Stores the trained model checkpoints.
- **src/**: Contains the core functionality separated by concern:
  - **data_processor.py**: Handles loading and preprocessing of text data
  - **trainer.py**: Manages the training process
  - **generator.py**: Provides text generation capabilities
- **notebooks/**: Jupyter notebooks for experimentation and visualization.
- **.cursor/rules/**: Contains Cursor IDE rules for AI assistance.
- **.scripts/**: Contains utility scripts like update_structure.sh.

## Architecture

BlogWriter follows a modular architecture with separation of concerns:
1. Data processing: Handles loading, preprocessing, and batching of text data
2. Model definition: Configures and initializes the language model for fine-tuning
3. Training loop: Manages the training process, metrics, and checkpoints
4. Generation: Provides text generation capabilities using the fine-tuned model
EOF

echo "Project structure has been updated in .cursor/rules/structure.mdc" 