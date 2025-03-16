import torch
import gradio as gr
from pathlib import Path
import os

from model import build_transformer
from dataset import causal_mask
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer

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

def translate(input_text):
    # Load the configuration
    config = get_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
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

def main():
    # Create Gradio interface
    demo = gr.Interface(
        fn=translate,
        inputs=gr.Textbox(
            lines=5, 
            placeholder="Enter English text to translate to Italian...",
            label="English Text"
        ),
        outputs=gr.Textbox(label="Italian Translation"),
        title="English to Italian Transformer Translation",
        description=("This app uses a Transformer model to translate English text to Italian. "
                    "If you haven't trained a model yet, please run 'python train_wb.py' first.\n\n"
                    "⚠️ *Translation Disclaimer* ⚠️\n\n"
                    "🔥 Warning: This translator might occasionally confuse 'pasta' with 'post office' due to limited GPU training. 🔥\n"
                    "Our model was trained on a GPU that was sweating harder than an Italian chef in a hot kitchen! 🍝💦\n"
                    "If your translation sounds like it was done by a tourist with a phrasebook and three espressos, that's why! 😂\n"
                    "We promise it's not the algorithm's fault - it's just dreaming of better cooling fans and more VRAM! 🌬️💾"),
        examples=[
            ["Hello, how are you today?"],
            ["I love machine learning and natural language processing."],
            ["The weather is beautiful today."]
        ]
    )
    
    # Launch the app
    demo.launch(share=True)

if __name__ == "__main__":
    main()
