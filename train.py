import torch
from torch.cuda import initial_seed
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchmetrics

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_weights_file_path, get_config, latest_weights_file_path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE    # Using BPE tokenizer as per first file
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
import os
import warnings

from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every token
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        # Create mask for decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)
        
        # Calculate output of decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get next token probabilities
        prob = model.project(out[:, -1])  # Only need the last token prediction
        
        # Select token with highest probability
        _, next_token = torch.max(prob, dim=1)
        
        # Append next token to decoder input
        decoder_input = torch.cat(
            [decoder_input, next_token.unsqueeze(0).unsqueeze(0)],
            dim=1
        )
        
        # Break if eos token is predicted
        if next_token.item() == eos_idx:
            break
            
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Ensure batch size is 1 for validation
            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-' * console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {model_out_text}')
            
            if count == num_examples:
                print_msg('-' * console_width)
                break
                
    # Calculate BLEU score
    bleu = torchmetrics.functional.bleu_score(predicted, expected)
    writer.add_scalar("validation/bleu", bleu, global_step)
    writer.flush()
    
    return bleu

def train_model(config):
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Using device: {device}")
    
    # Create model folder
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Get datasets and tokenizers
    train_dataloader, val_dataloader, tokenizer_tgt, tokenizer_src = get_ds(config)
    
    # Print vocabulary sizes
    print(f"Source vocabulary size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocabulary size: {tokenizer_tgt.get_vocab_size()}")
    
    # Initialize model
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)
    
    # Initialize tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Initialize learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=len(train_dataloader),
        epochs=config['num_epochs'],
        pct_start=0.1  # 10% warmup
    )
    
    # Load checkpoint if specified
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    
    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            # Get batch data
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )
            proj_output = model.project(decoder_output)
            
            # Calculate loss
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{loss.item():6.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            
            # Log to tensorboard
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.flush()
            
            global_step += 1
        
        # Run validation
        val_bleu = run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config['seq_len'],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer
        )
        
        # Save checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'val_bleu': val_bleu
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)