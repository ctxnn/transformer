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

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every token we get from the decoder 
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the SOS token 
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break 
        
        # Build mask for target (note: using source_mask type for consistency)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)   # (b, seq_len, d_model)
        # Next token 
        prob = model.project(out[:, -1])  # (b, tgt_vocab_size)
        # Select the token with the highest probability
        _, next_token = torch.max(prob, dim=1)      # (b,)
        # **IMPORTANT**: you probably intended to cat the new token, not `[SOS]` again
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1).to(device)
        
        if next_token.item() == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval() 
    count = 0 
    
    source_texts = []
    expected = []
    predicted = []
    
    # size of the control window 
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # check that the batch size is 1
            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation."   
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and predicted texts
            print_msg('-' * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")
            
            if count == num_examples:
                print_msg('-' * console_width)
                break

    # Compute additional metrics using torchmetrics
    # Compute character error rate
    cer_metric = torchmetrics.CharErrorRate()
    cer = cer_metric(predicted, expected)
    writer.add_scalar("validation/cer", cer, global_state)
    writer.flush()

    # Compute word error rate
    wer_metric = torchmetrics.WordErrorRate()
    wer = wer_metric(predicted, expected)
    writer.add_scalar("validation/wer", wer, global_state)
    writer.flush()
    
    # Compute BLEU score
    bleu = torchmetrics.functional.bleu_score(predicted, expected)
    writer.add_scalar("validation/bleu", bleu, global_state)
    writer.flush()

def get_all_sentences(ds, lang):
    """
    Yields text sentences one by one from a dataset for a specific language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Builds (or loads) a BPE tokenizer for the specified language. 
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    # IMPORTANT: We need to use the Tokenizer class from `tokenizers`
    # if the file does not exist, we build and save a new tokenizer
    if not tokenizer_path.exists():
        # Create a new Tokenizer instance with BPE model
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
        # Train using an iterator
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # If the tokenizer file exists, load it
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build the tokenizer (using BPE as specified)
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% of the data for training
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create datasets for training and validation
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Calculate the maximum sequence lengths in both source and target languages
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max len src: {max_len_src}, max len tgt: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # Notice the return order changed to keep it consistent with the usage
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src, vocab_tgt):
    model = build_transformer(
        vocab_src,
        vocab_tgt,
        config['seq_len'],
        config['seq_len'],
        config['d_model']
    )
    return model

def train_model(config):
    # define the device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Device memory: {props.total_memory / 1024**3:.2f} GB")
    elif device == "mps":
        print("Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
    
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # TensorBoard writer
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config.get('preload', None)
    model_filename = None
    if preload:
        if preload == 'latest':
            model_filename = latest_weights_file_path(config)
        else:
            model_filename = get_weights_file_path(config, preload)
    if model_filename:
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # Make sure to load the model as well
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")
    
    # Use tokenizer_src’s PAD token ID for the loss ignore index
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    # (b, 1, seq_len, seq_len)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)               # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask,
                                          decoder_input, decoder_mask)               # (b, seq_len, d_model)
            proj_output = model.project(decoder_output)                              # (b, seq_len, tgt_vocab_size)
            
            label = batch['label'].to(device)  # (b, seq_len)

            # Flatten for CrossEntropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1

        # Validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg),
                       global_step, writer)

        # Save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        Path(model_folder).mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)