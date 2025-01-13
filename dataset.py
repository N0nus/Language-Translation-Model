import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

def load_csv_data(src_file: str, tgt_file: str):
    """Load source and target data from CSV files"""
    src_data = pd.read_csv(src_file, encoding='utf-8', header=None)[0].tolist()
    tgt_data = pd.read_csv(tgt_file, encoding='utf-8', header=None)[0].tolist()
    return list(zip(src_data, tgt_data))

class TranslationDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        # Filter out sentences that are too long
        self.filtered_ds = []
        for src_text, tgt_text in ds:
            # Transform the text into tokens
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

            # Check if the sentences are too long
            if len(enc_input_tokens) <= seq_len - 2 and len(dec_input_tokens) <= seq_len - 2:  # -2 for SOS and EOS tokens
                self.filtered_ds.append((src_text, tgt_text))

    def __len__(self):
        return len(self.filtered_ds)

    def __getitem__(self, idx):
        src_text, tgt_text = self.filtered_ds[idx]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def get_all_sentences(ds, lang_index):
    """Get all sentences for tokenizer training"""
    # Since ds contains tuples of (src, tgt), we can directly index them
    for item in ds:
        yield item[lang_index]
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0