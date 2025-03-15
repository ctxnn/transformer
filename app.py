import gradio as gr
import torch
from model import build_transformer
from dataset import causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from tokenizers import Tokenizer
from pathlib import Path

def translate(text, model_path):
    try:
        if not text.strip():
            return "Please enter some text to translate"
        
        print("\n=== Starting Translation Process ===")
        print(f"Input text: {text}")
        print(f"Model path: {model_path}")
            
        if not Path(model_path).exists():
            error = f"Error: Model file not found at {model_path}. Please check the path and make sure the model file exists."
            print(error)
            return error
            
        # Load config
        config = get_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizers
        try:
            tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
            tokenizer_tgt = Tokenizer.from_file("tokenizer_it.json")
            print("Loaded tokenizers successfully")
            print(f"Source vocab size: {tokenizer_src.get_vocab_size()}")
            print(f"Target vocab size: {tokenizer_tgt.get_vocab_size()}")
        except Exception as e:
            error = f"Error loading tokenizers: {str(e)}"
            print(error)
            return error
        
        # Load the model
        try:
            # First load the state dict to check its structure
            state = torch.load(model_path, map_location=device)
            print(f"Loaded state dict with keys: {list(state.keys())}")
            
            if not isinstance(state, dict):
                error = "Model file is not in the correct format"
                print(error)
                return error
                
            # Build the model with the same configuration as training
            model = build_transformer(
                tokenizer_src.get_vocab_size(),
                tokenizer_tgt.get_vocab_size(),
                config['seq_len'],
                config['seq_len'],
                d_model=config['d_model']
            ).to(device)
            print("Built transformer model")
            
            # Load the state dict
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                error = "Model file does not contain model_state_dict"
                print(error)
                return error
                
            model.eval()
            print("Loaded model weights successfully")
            
        except Exception as e:
            error = f"Error loading model: {str(e)}"
            print(error)
            return error
        
        try:
            # Tokenize input text
            source_tokens = tokenizer_src.encode(text)
            source = torch.tensor([source_tokens.ids]).to(device)
            print(f"Input text tokenized: {source.shape}")
            print(f"Input tokens: {source_tokens.tokens}")
            
            # Create source mask
            source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).to(device)
            
            # Translate
            sos_idx = tokenizer_tgt.token_to_id('[SOS]')
            eos_idx = tokenizer_tgt.token_to_id('[EOS]')
            pad_idx = tokenizer_tgt.token_to_id('[PAD]')
            
            with torch.no_grad():
                # Get encoder output
                encoder_output = model.encode(source, source_mask)
                decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
                
                # Temperature for softmax
                temperature = 0.7
                
                output_tokens = []
                for i in range(config['seq_len']):
                    # Create decoder mask
                    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
                    
                    # Get model output
                    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
                    prob = model.project(out[:, -1])
                    
                    # Apply temperature
                    prob = torch.div(prob, temperature)
                    
                    # Get next token
                    next_word = torch.argmax(prob, dim=1)
                    next_word_item = next_word.item()
                    
                    # Break if we hit EOS or PAD
                    if next_word_item in [eos_idx, pad_idx]:
                        break
                        
                    output_tokens.append(next_word_item)
                    decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
                    print(f"Generated token {i}: {next_word_item}")
                    
            print(f"Generated token ids: {output_tokens}")
            
            # Convert tokens to text
            if not output_tokens:
                return "Model generated empty translation"
            
            # Decode tokens to text
            translated_text = tokenizer_tgt.decode(output_tokens)
            print(f"Raw translation: {translated_text}")
            
            # Clean the output
            translated_text = translated_text.replace('[PAD]', '').replace('[SOS]', '').replace('[EOS]', '').strip()
            translated_text = ' '.join(translated_text.split())  # Remove extra spaces
            
            if not translated_text.strip():
                return "Translation resulted in empty text"
            
            print(f"Final translation: {translated_text}")
            print("=== Translation Process Complete ===\n")
            return translated_text
            
        except Exception as e:
            error = f"Error during translation process: {str(e)}"
            print(error)
            return error
        
    except Exception as e:
        import traceback
        error_msg = f"Error during translation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

# Get latest model path
config = get_config()
default_model = latest_weights_file_path(config)
if not default_model:
    default_model = "opus_books_weights/mach_trans_model_29.pt"

# Create Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter text to translate (English to Italian)", placeholder="Type your English text here..."),
        gr.Textbox(
            label="Model path", 
            value=default_model,
            placeholder="Path to your trained model file"
        )
    ],
    outputs=gr.Textbox(label="Italian Translation"),
    title="English to Italian Neural Machine Translation",
    description="Translate English text to Italian using a Transformer model.",
    examples=[
        ["Hello, how are you?", default_model],
        ["What is your name?", default_model],
        ["I love learning new languages.", default_model]
    ],
    theme="default"
)

if __name__ == "__main__":
    print("Starting translation app...")
    print(f"Default model path: {default_model}")
    iface.launch(share=True)
