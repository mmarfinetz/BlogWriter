# BlogWriter

A custom GPT fine-tuning project to generate blog posts in your personal writing style.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your training data in the `data` folder
3. Run the training script: `python train.py`
4. Generate new content: `python generate.py`

## Using the Generator

You can generate blog posts in several ways:

1. **Interactive Mode**: Run `python generate.py --model models/your_model` and you'll be prompted to enter text.

2. **Command Line Prompt**: Directly provide a prompt via command line:
   ```
   python generate.py --model models/your_model --prompt "Start writing about..."
   ```

3. **Using Prompt Files**: For longer prompts, save them in a text file and use:
   ```
   python generate.py --model models/your_model --prompt_file prompts/your_prompt.txt
   ```

## Project Structure

- `data/`: Place your writing samples here
- `models/`: Saved fine-tuned models
- `src/`: Source code for training and generation
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `prompts/`: Text files containing prompts for generation