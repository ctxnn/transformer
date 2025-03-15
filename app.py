import gradio as gr
import torch
from model import build_transformer
from dataset import causal_mask
from config import get_config, get_weights_file_path
from tokenizers import Tokenizer
from pathlib import Path

def translate(text, model_path):
    # Load config
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    # Load the model
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                            config['seq_len'], config['seq_len'], config['d_model']).to(device)
    
    # Load the weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Tokenize input text
    source_tokens = tokenizer_src.encode(text)
    source = torch.tensor([source_tokens.ids]).to(device)
    
    # Create source mask
    source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).to(device)
    
    # Translate
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == config['seq_len']:
            break
            
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
        
        if next_word == eos_idx:
            break
            
    # Convert tokens to text
    translated_tokens = decoder_input.squeeze(0).detach().cpu().numpy()
    translated_text = tokenizer_tgt.decode(translated_tokens)
    
    return translated_text

# Create Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter text to translate"),
        gr.Textbox(label="Model path (.pt file)", value="weights/tmodel_0.pt")
    ],
    outputs=gr.Textbox(label="Translation"),
    title="Neural Machine Translation",
    description=f"Translate text using a Transformer model"
)

if __name__ == "__main__":
    iface.launch(share=True, enable_queue=True)
