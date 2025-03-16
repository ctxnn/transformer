# Transformer Machine Translation

![Transformer Architecture](image.png)

## Project Overview
This project implements a Transformer-based machine translation system that translates English text to Italian. It's built using PyTorch and follows the architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Performance Observations
- Training on Apple Silicon 'mps' backend is very slow for transformers
- Training progress that took overnight (8+ hours) on MPS to reach 3% of one epoch was completed in just 3 seconds on an NVIDIA L4 GPU
- Using a 4x CPU setup on Lightning AI worked much better than a single MPS GPU backend

## Transformer Architecture

The transformer model in this project follows the original architecture with:

### Encoder
- **Multi-head Self-Attention**: Allows the model to focus on different parts of the input sequence
- **Position-wise Feed-Forward Networks**: Processes the attention output through fully connected layers
- **Layer Normalization**: Stabilizes the learning process
- **Residual Connections**: Helps with gradient flow during training

### Decoder
- **Masked Multi-head Self-Attention**: Prevents the model from looking at future tokens
- **Multi-head Cross-Attention**: Attends to the encoder's output
- **Position-wise Feed-Forward Networks**: Same as in the encoder
- **Layer Normalization and Residual Connections**: Same as in the encoder

### Other Components
- **Positional Encoding**: Adds information about the position of tokens in the sequence
- **Input/Output Embeddings**: Converts tokens to vectors and vice versa
- **Linear Projection and Softmax**: Converts decoder output to probability distribution over vocabulary

## Project Structure

### Core Files
- **`model.py`**: Implements the complete Transformer architecture including multi-head attention, positional encoding, encoder/decoder stacks, and helper functions.
- **`dataset.py`**: Handles data processing with the BilingualDataset class and causal masking for the decoder.
- **`config.py`**: Contains configuration parameters and functions to manage model weights paths.
- **`train_wb.py`**: Implements the training loop with Weights & Biases integration, validation, and metrics tracking.
- **`inference.py`**: Provides a Gradio interface for interactive translation with support for both trained models and demo mode.
- **`app.py`**: Simplified version of inference.py configured for deployment, always running in demo mode by default.

### Additional Files
- **`requirements.txt`**: Lists Python dependencies for local development.
- **`requirements_hf.txt`**: Lists dependencies for deployment.
- **`.gitignore`**: Specifies files to exclude from version control.

## Model Parameters
The default configuration uses:
- Embedding dimension: 512
- Number of encoder/decoder layers: 6
- Number of attention heads: 8
- Feed-forward dimension: 2048
- Dropout rate: 0.1
- Maximum sequence length: 350

## Training Process
The model is trained on the OPUS Books dataset, which contains parallel text in English and Italian. The training process includes:
1. Tokenization of source and target text
2. Creation of source and target embeddings
3. Forward pass through the transformer
4. Calculation of loss using cross-entropy
5. Backpropagation and optimization
6. Validation using metrics like BLEU, Character Error Rate, and Word Error Rate

## Inference
During inference, the model:
1. Encodes the source sentence
2. Generates the target sentence token by token using greedy decoding
3. Stops when it generates an EOS token or reaches maximum length

## Running Locally
To run the translation app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the inference app
python inference.py

# To run in demo mode (no model required)
python inference.py --demo
```

## Training Your Own Model
To train your own translation model:

```bash
# Run the training script
python train_wb.py
```

Note: Training requires significant computational resources. For best results, use a GPU.
