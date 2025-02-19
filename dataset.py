import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[SOS]")]).long()
        self.eos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[EOS]")]).long()
        self.pad_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[PAD]")]).long()



    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        src_encodings = self.tokenizer_src.encode(src_text).ids #it will give us the token ids of the source text corresponding to each word and build an array
        tgt_encodings = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(src_encodings) - 2 # -2 for sos and eos tokens
        dec_num_padding_tokens = self.seq_len - len(tgt_encodings) - 2 # -2 for sos and eos tokens

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too short")

        # add sos and eos to the source text and pad it
        encoder_input = torch.cat([
                                   self.sos_token,
                                   torch.Tensor(src_encodings),
                                   self.eos_token,
                                   self.pad_token.repeat(enc_num_padding_tokens)
            ]
        ).long()

        # add sos to the target text and pad it
        decoder_input = torch.cat([
                                      self.sos_token,
                                      torch.Tensor(tgt_encodings),
                                      self.pad_token.repeat(dec_num_padding_tokens)
                ]
          ).long()

        # add eos to the target text and pad it(we expect output from the decoder)
        label = torch.cat([
                                torch.Tensor(tgt_encodings),
                                self.eos_token,
                                self.pad_token.repeat(dec_num_padding_tokens)
                ]
            ).long()

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            # we only want the decoder to pay attention to the tokens that are not padding tokens in the self attention mechanism thats why we apply a mask
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            # in decoder we need a special mask where each words can only pay attention to the words that came before it and the padding thing too
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_len) & (1, seq_len, seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0
