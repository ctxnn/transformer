from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb

import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, temperature=1.0):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    comma_idx = tokenizer_tgt.token_to_id(',') if ',' in tokenizer_tgt.get_vocab() else -1
    
    # Debug: Print vocabulary to check token distributions
    print(f"Vocabulary size: {tokenizer_tgt.get_vocab_size()}")
    print(f"SOS token ID: {sos_idx}, EOS token ID: {eos_idx}, Comma token ID: {comma_idx}")
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # Keep track of outputs and explicitly banned tokens
    banned_tokens = set()
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token with temperature
        logits = model.project(out[:, -1])
        
        # Debug: Print raw logits for the first few tokens to see distribution
        if decoder_input.size(1) < 5:
            top_logits, top_indices = torch.topk(logits, 10)
            print(f"Step {decoder_input.size(1)} - Top logits: {top_logits}")
            print(f"Step {decoder_input.size(1)} - Top tokens: {[tokenizer_tgt.id_to_token(idx.item()) for idx in top_indices[0]]}")
        
        # If comma is heavily favored, artificially reduce its probability
        if comma_idx != -1:
            penalty_factor = 100.0  # Large penalty to discourage commas
            if decoder_input.size(1) > 2 and comma_idx in banned_tokens:
                logits[0, comma_idx] = -1e9  # Effectively ban commas after seeing a few
            else:
                # Apply penalty to comma token
                logits[0, comma_idx] = logits[0, comma_idx] / penalty_factor
        
        # Add noise for exploration
        noise = (torch.rand_like(logits) * 0.05)  # 5% noise
        logits = logits + noise
        
        # Apply temperature scaling
        logits = logits / max(0.7, temperature)  # Ensure minimum temperature for diversity
        
        # Apply softmax to convert to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Use nucleus sampling (top-p) instead of just temperature
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold (top-p)
        sorted_indices_to_remove = cumulative_probs > 0.9  # Use top 90% of probability mass
        sorted_indices_to_remove[..., 0] = False  # Keep the top token
        
        # Scatter sorted indices to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('Inf')
        
        # Sample from filtered distribution
        next_token_logits = filtered_logits / max(0.8, temperature)  # Additional temperature control
        next_token_probs = torch.softmax(next_token_logits, dim=1)
        next_word = torch.multinomial(next_token_probs, num_samples=1).item()
        
        # If we're still getting commas, force different tokens
        if next_word == comma_idx and comma_idx in banned_tokens:
            # Get highest probability non-comma token
            next_probs = next_token_probs.clone()
            next_probs[0, comma_idx] = 0.0  # Zero out comma probability
            next_word = torch.multinomial(next_probs, num_samples=1).item()
        
        # If we've seen too many commas, ban them
        last_tokens = [t.item() for t in decoder_input[0, -3:]] if decoder_input.size(1) > 3 else []
        if next_word == comma_idx and comma_idx in last_tokens:
            banned_tokens.add(comma_idx)
            
            # Get highest probability non-comma token
            next_probs = next_token_probs.clone()
            next_probs[0, comma_idx] = 0.0  # Zero out comma probability
            next_word = torch.multinomial(next_probs, num_samples=1).item()
        
        # Debug output
        print(f"Selected token: {next_word} ({tokenizer_tgt.id_to_token(next_word) if next_word < tokenizer_tgt.get_vocab_size() else 'UNK'})")
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=3):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    
    # Initialize with a single beam with the start token
    beams = [{'sequence': torch.LongTensor([[sos_idx]]).to(device),
              'score': 0.0,
              'is_complete': False}]
    
    for _ in range(max_len - 1):
        new_beams = []
        # For each existing beam
        for beam in beams:
            if beam['is_complete']:
                new_beams.append(beam)
                continue
                
            decoder_input = beam['sequence']
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            log_probs = torch.log_softmax(prob, dim=1)
            
            topk_probs, topk_ids = torch.topk(log_probs, beam_size, dim=1)
            
            for i in range(beam_size):
                new_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                new_score = beam['score'] + topk_probs[0, i].item()
                new_sequence = torch.cat([decoder_input, new_id], dim=1)
                is_complete = (new_id.item() == eos_idx)
                
                new_beams.append({
                    'sequence': new_sequence,
                    'score': new_score,
                    'is_complete': is_complete
                })
        
        # Keep only the top beam_size beams
        beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
        
        # Check if all beams are complete
        if all(beam['is_complete'] for beam in beams):
            break
    
    # Return the highest scoring complete beam, or the highest scoring beam if none are complete
    for beam in beams:
        if beam['is_complete']:
            return beam['sequence'].squeeze(0)
    return beams[0]['sequence'].squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    
    # Add debug information
    print_msg("Starting validation with token distribution analysis...")
    
    # Check tokenizer for comma frequency
    if ',' in tokenizer_tgt.get_vocab():
        comma_id = tokenizer_tgt.token_to_id(',')
        print_msg(f"Comma token ID: {comma_id}")
    else:
        print_msg("Comma is not a separate token in the vocabulary")
    
    # Print some vocabulary information
    vocab = tokenizer_tgt.get_vocab()
    print_msg(f"Vocabulary size: {len(vocab)}")
    print_msg(f"Sample tokens: {list(vocab.items())[:20]}")

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            # Try decoding with high temperature for more diversity
            print_msg(f"=== Decoding example {count} with temperature=1.2 ===")
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, temperature=1.2)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Detailed token analysis
            tokens = model_out.detach().cpu().numpy()
            token_strs = [tokenizer_tgt.id_to_token(t) if t < tokenizer_tgt.get_vocab_size() else "UNK" for t in tokens]
            token_analysis = [(i, t, s) for i, (t, s) in enumerate(zip(tokens, token_strs))]
            print_msg(f"Token analysis: {token_analysis}")
            
            # Count token frequencies
            from collections import Counter
            token_counter = Counter(tokens)
            print_msg(f"Token frequency: {token_counter.most_common(5)}")

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def get_latest_checkpoint(folder_path):
    """Get the latest checkpoint file from the given folder."""
    import os
    import glob
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Checkpoint folder {folder_path} does not exist.")
        return None
    
    # Get all weight files
    weight_files = glob.glob(os.path.join(folder_path, "*.pt"))
    
    if not weight_files:
        print(f"Warning: No checkpoint files found in {folder_path}")
        return None
    
    # Sort by modification time (latest first)
    latest_file = max(weight_files, key=os.path.getmtime)
    
    # Extract just the epoch number from the filename
    import re
    match = re.search(r'(\d+)\.pt$', latest_file)
    if match:
        epoch_number = match.group(1)
        return epoch_number
    
    # If we can't extract the epoch number, return the full filename
    return os.path.basename(latest_file)

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Hardcode checkpoint folder and find the latest file
    checkpoint_folder = "/teamspace/studios/this_studio/transformer/weights"
    latest_checkpoint = get_latest_checkpoint(checkpoint_folder)
    if latest_checkpoint:
        config['preload'] = latest_checkpoint
        print(f"Automatically loading latest checkpoint: {latest_checkpoint}")

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        import os

        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    wandb.init(
        # set the wandb project where this run will be logged
        project="machine-translation-transformer",
        
        # track hyperparameters and run metadata
        config=config
    )
    
    train_model(config)