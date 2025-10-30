"""
Dataset wrapper for the HuggingFace Shakespeare dataset.
Provides a PyTorch Dataset + collate_fn that returns encoder inputs and decoder labels.
"""

from typing import Optional
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from .preprocess import encode_encoder, encode_decoder, decoder_tokenizer
from transformers import RobertaTokenizer

# Initialize encoder tokenizer globally to avoid re-instantiating
encoder_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class ProseDataset(Dataset):
    """
    Wrapper around Roudranil/shakespearean-and-modern-english-conversational-dataset.
    Each item returns:
      - encoder inputs (roberta) for the Shakespeare text (source)
      - decoder inputs & labels (mBART) for the modern English text (target)
    """

    def __init__(self, split: str = "train", max_length: int = 256, limit: Optional[int] = None):
        self.max_length = max_length
        # load HF dataset split
        ds = load_dataset("Roudranil/shakespearean-and-modern-english-conversational-dataset", split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        # store raw pairs
        columns = ds.column_names
        print("Available columns:", columns)

        # Map columns correctly for this dataset
        shakes_col = "og_response"           # Shakespeare text (source)
        modern_col = "translated_dialog"     # Modern English (target)

        self.pairs = [(row[shakes_col], row[modern_col]) for row in ds]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return {"src": src, "tgt": tgt}

    @staticmethod
    def collate_fn(batch, max_length=256):
        """Takes a list of items from __getitem__ and returns batched tensors."""
        src_texts = [b["src"] for b in batch]
        tgt_texts = [b["tgt"] for b in batch]

        # Encode encoder inputs (Roberta) with truncation and padding
        enc = encoder_tokenizer(
            src_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Encode decoder inputs (mBART) with truncation and padding
        dec = decoder_tokenizer(
            tgt_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        enc["input_ids"] = enc["input_ids"].clamp(0, encoder_tokenizer.vocab_size - 1)
        dec["input_ids"] = dec["input_ids"].clamp(0, decoder_tokenizer.vocab_size - 1)

        # Prepare labels: replace padding token id with -100 to ignore in loss
        labels = dec["input_ids"].clone()
        labels[labels == decoder_tokenizer.pad_token_id] = -100

        batch_out = {
            "enc_input_ids": enc["input_ids"],
            "enc_attention_mask": enc["attention_mask"],
            "dec_input_ids": dec["input_ids"],
            "dec_attention_mask": dec["attention_mask"],
            "labels": labels
        }

        # DEBUG: i wrote this for debugging purposes
        #("--- Batch shapes ---")
        #print("Encoder input_ids:", enc["input_ids"].shape)
        #print("Encoder attention_mask:", enc["attention_mask"].shape)
        #print("Decoder input_ids:", dec["input_ids"].shape)
        #print("Decoder attention_mask:", dec["attention_mask"].shape)
        #print("Labels:", labels.shape)

        # Token ID sanity
        #print("--- Token ID sanity ---")
        #print("Encoder max ID:", enc["input_ids"].max().item(), "Vocab size:", encoder_tokenizer.vocab_size)
        #print("Decoder max ID:", dec["input_ids"].max().item(), "Vocab size:", decoder_tokenizer.vocab_size)
        #print("Encoder pad token:", encoder_tokenizer.pad_token_id)
        #print("Decoder pad token:", decoder_tokenizer.pad_token_id)


        return batch_out
