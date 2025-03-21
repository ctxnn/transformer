import torch
import gradio as gr
from pathlib import Path
import os
import argparse

from model import build_transformer
from dataset import causal_mask
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def create_dummy_tokenizer(vocab_size=1000):
    """Create a simple tokenizer for demo purposes"""
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Create a basic vocabulary
    vocab = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2, "[EOS]": 3}
    
    # Add some common words to the vocabulary
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "hello", "world", "machine", "learning", "translation", "model", "transformer",
        "language", "processing", "natural", "artificial", "intelligence", "neural", "network"
    ]
    
    for i, word in enumerate(common_words, start=len(vocab)):
        vocab[word] = i
    
    # Fill the rest with dummy tokens
    for i in range(len(vocab), vocab_size):
        vocab[f"token{i}"] = i
    
    tokenizer.model = WordLevel(vocab, unk_token="[UNK]")
    return tokenizer

def translate(input_text, use_demo_mode=True):
    """Translate text using either a trained model or a demo model"""
    # Load the configuration
    config = get_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_demo_mode:
        # Create dummy tokenizers for demo
        tokenizer_src = create_dummy_tokenizer()
        tokenizer_tgt = create_dummy_tokenizer()
        
        # Create a dummy model
        model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            config['seq_len'],
            config['seq_len'],
            d_model=config['d_model']
        ).to(device)
        
        # For demo mode, just return a predefined response
        return f"[DEMO MODE] Translation of: '{input_text}'\n\nCiao, questa è una traduzione di esempio. Il modello reale non è stato caricato."
    
    # For real model mode:
    # Check if tokenizers exist
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        return "Tokenizer files not found. Please run the training script first to generate tokenizers, or use demo mode."
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Load the model
    model_path = latest_weights_file_path(config)
    if model_path is None:
        return "No trained model found. Please train the model first by running 'python train_wb.py', or use demo mode."
    
    print(f"Loading model from: {model_path}")
    
    # Check if the model file exists
    if not Path(model_path).exists():
        return f"Model file not found at: {model_path}. Please check the path and ensure the model exists."
    
    # Load the trained model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'], 
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)
    
    try:
        # Load model weights
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        return f"Error loading model: {str(e)}"
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize the input text
    encoder_input = tokenizer_src.encode(input_text).ids
    
    # Make sure the sequence is not too long
    if len(encoder_input) > config['seq_len'] - 2:  # -2 for SOS and EOS tokens
        encoder_input = encoder_input[:config['seq_len'] - 2]
    
    # Add SOS and EOS tokens
    sos_token = tokenizer_src.token_to_id('[SOS]')
    eos_token = tokenizer_src.token_to_id('[EOS]')
    pad_token = tokenizer_src.token_to_id('[PAD]')
    
    # Add SOS and EOS tokens and padding
    encoder_input = [sos_token] + encoder_input + [eos_token]
    encoder_padding = [pad_token] * (config['seq_len'] - len(encoder_input))
    encoder_input = encoder_input + encoder_padding
    
    # Convert to tensor
    encoder_input = torch.tensor(encoder_input).unsqueeze(0).to(device)  # (1, seq_len)
    
    # Create source mask
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)  # (1, 1, 1, seq_len)
    
    # Perform translation
    model_output = greedy_decode(
        model, 
        encoder_input, 
        encoder_mask, 
        tokenizer_src, 
        tokenizer_tgt, 
        config['seq_len'],
        device
    )
    
    # Decode the output
    translation = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
    
    # Clean up the translation by removing special tokens and extra spaces
    translation = translation.replace("[SOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
    
    return translation

# Create Gradio interface
demo = gr.Interface(
    fn=lambda text: translate(text, use_demo_mode=True),  # Always use demo mode for Hugging Face Space
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Enter English text to translate to Italian...",
        label="English Text"
    ),
    outputs=gr.Textbox(label="Italian Translation"),
    title="English to Italian Transformer Translation",
    description=("This app uses a Transformer model to translate English text to Italian. "
                "⚠️ *Translation Disclaimer* ⚠️\n\n"
                "🔥 Warning: This translator might occasionally confuse 'pasta' with 'post office' due to limited GPU training. 🔥\n"
                "Our model was trained on a GPU that was sweating harder than an Italian chef in a hot kitchen! 🍝💦\n"
                "If your translation sounds like it was done by a tourist with a phrasebook and three espressos, that's why! 😂\n"
                "We promise it's not the algorithm's fault - it's just dreaming of better cooling fans and more VRAM! 🌬️💾\n\n"
                "Note: This is running in demo mode for Hugging Face Spaces. For full functionality, clone the repository and run locally."),
    examples=[
        ["Hello, how are you today?"],
        ["I love machine learning and natural language processing."],
        ["The weather is beautiful today."]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
