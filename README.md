# BlogWriter

A tool for generating blog posts using either:
1. A fine-tuned GPT-2 model customized to your writing style
2. The Anthropic Claude API

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Here's how to go from zero to generating content in your writing style:

1. **Prepare your writing samples**
   - Create a `data/` directory and add your writing samples as `.txt` files
   - The more samples you provide, the better the model will learn your style
   - Aim for at least 10,000 words of your writing in total

2. **Train your model with one command**
   ```bash
   python train.py --data_dir data --output_dir models/my_model --epochs 3
   ```

3. **Generate content**
   ```bash
   python generate.py --model models/my_model --prompt "Write a blog post about AI"
   ```

That's it! You now have a custom language model that writes like you.

## Training Details

### Option 1: Command Line Training (Recommended)

The `train.py` script handles everything - from processing your writing samples to training and saving the model:

```bash
python train.py \
  --data_dir data \
  --output_dir models/my_model \
  --model gpt2 \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --fp16
```

Parameters:
- `--data_dir`: Directory containing your .txt writing samples
- `--output_dir`: Where to save the fine-tuned model
- `--model`: Base model to fine-tune (`gpt2`, `gpt2-medium`, or `gpt2-large`)
- `--epochs`: Number of training epochs (2-5 recommended)
- `--batch_size`: Batch size for training (adjust based on your GPU memory)
- `--fp16`: Use mixed-precision training (faster on supported GPUs)
- `--max_length`: Maximum sequence length (default: 512)

Adjust training time:
- For faster training: use fewer epochs or the smaller `gpt2` model
- For better quality: use more epochs or larger models like `gpt2-medium`

Training a GPT-2 model typically takes:
- Small CPU: Several hours
- GPU: 15-60 minutes (depending on data size and GPU)

### Option 2: Jupyter Notebook Training

Alternatively, you can use the included Jupyter notebook for a more interactive experience:

1. Run through `notebooks/fine_tuning_tutorial.ipynb` step by step
2. The notebook provides more detailed explanations and visualizations
3. Good for learning how the training process works

## Using Your Trained Model

After training, use your model to generate content with the `generate.py` script:

```bash
python generate.py \
  --model models/my_model \
  --prompt_file prompts/my_prompt.txt \
  --temperature 0.7 \
  --length 1000 \
  --output my_blog_post.txt
```

Parameters:
- `--model`: Path to your fine-tuned model directory
- `--prompt_file`: Path to a text file containing the prompt
- `--prompt`: Directly provide a prompt string (alternative to prompt_file)
- `--length`: Maximum length of generated text
- `--temperature`: Controls randomness (0.0-1.0, higher = more creative)
- `--top_k`: Number of top tokens to consider (default: 50)
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--repetition_penalty`: Penalizes repetition (default: 1.2)
- `--style_aware`: Enable style-aware generation
- `--style_strength`: How strongly to enforce style (0-1)
- `--output`: Output file path

For advanced evaluation:
```bash
python generate.py \
  --model models/my_model \
  --prompt_file prompts/my_prompt.txt \
  --evaluate \
  --references data/ \
  --eval_output evaluation_report.txt \
  --visualize
```

## Using Claude API (Alternative)

If you prefer to use the Anthropic Claude API:

1. Set your API key as an environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your-api-key
   ```

2. Use the `generate_claude.py` script:
   ```bash
   python generate_claude.py \
     --prompt_file prompts/my_prompt.txt \
     --temperature 0.7 \
     --max_tokens 4000 \
     --output my_blog_post.txt
   ```

Parameters:
- `--prompt_file`: Path to a text file containing the prompt
- `--model`: Claude model to use (default: claude-3-5-sonnet-20240620)
- `--temperature`: Controls randomness (0.0-1.0)
- `--max_tokens`: Maximum length of generated text
- `--output`: Output file path

## Examples

### Create a prompt

```bash
echo "Write a blog post about the impact of artificial intelligence on modern workplace productivity" > prompts/ai_workplace.txt
```

### Generate with your custom model

```bash
python generate.py --model models/my_model --prompt_file prompts/ai_workplace.txt --output ai_blog.txt
```

### Generate with Claude (alternative)

```bash
python generate_claude.py --prompt_file prompts/ai_workplace.txt --output ai_blog_claude.txt
```

## Template-Based Generation

The template-based generation system allows for more controlled content creation:

- **Content Types**: Blog posts, essays, technical articles, newsletters
- **Writing Styles**: Formal, casual, academic, conversational, professional
- **Tones**: Informative, persuasive, entertaining, authoritative, friendly, analytical
- **Formats**: Standard, listicle, guide, review, opinion, comparison

Templates can be combined with few-shot examples from your training data to guide the generated content and maintain consistent style throughout longer pieces.

See the `test_template_generation.py` script for a complete example of template-based generation.

## External API Integration

The project supports multiple language models through API integrations:

- **OpenAI GPT Models**: Generate content using models like GPT-4 via `generate_openai.py`
- **Claude Models**: Generate content using Anthropic's Claude 3.5 via `generate_claude.py`

These integrations require setting up appropriate API keys in your environment.

## Project Structure

- `data/`: Place your writing samples here
- `models/`: Saved fine-tuned models
- `src/`: Source code for training and generation
  - `templates.py`: Template definitions for different content types
  - `generator.py`: Content generation functionality with style controls
  - `style_config.py`: Style parameters and configuration
  - `data_processor.py`: Data processing utilities for training and examples
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `prompts/`: Text files containing prompts for generation