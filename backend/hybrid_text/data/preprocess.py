"""
Research-grade preprocessing utilities.

This module performs:
 - light normalization
 - tokenizer initialization for encoder (RoBERTa) and decoder (mBART)
 - helper functions to convert text -> token tensors (with padding/truncation)
"""

import re
import unicodedata
from typing import List, Dict, Any
from transformers import RobertaTokenizerFast, MBart50Tokenizer

# Choose robust tokenizers for English
ENCODER_MODEL = "roberta-base"
DECODER_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
DECODER_LANG = "en_XX"

# Initialize tokenizers once 
encoder_tokenizer = RobertaTokenizerFast.from_pretrained(ENCODER_MODEL)
decoder_tokenizer = MBart50Tokenizer.from_pretrained(DECODER_MODEL)
decoder_tokenizer.src_lang = DECODER_LANG
decoder_tokenizer.tgt_lang = DECODER_LANG

def normalize_text(text: str) -> str:
    """Unicode normalize, collapse whitespace and basic cleaning."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    # reduce multiple whitespace to single
    text = re.sub(r"\s+", " ", text)
    return text

def encode_encoder(texts: List[str], max_length: int = 256) -> Dict[str, Any]:
    """Tokenize for encoder (RoBERTa). Returns dict of input_ids and attention_mask (torch tensors)."""
    texts = [normalize_text(t) for t in texts]
    out = encoder_tokenizer(
        texts,
        padding="longest",  
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return out

def encode_decoder(texts: List[str], max_length: int = 256) -> Dict[str, Any]:
    """Tokenize for decoder (mBART). Returns input_ids and attention_mask (torch tensors)."""
    texts = [normalize_text(t) for t in texts]
    out = decoder_tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return out

def detokenize_decoder(token_ids):
    """Convert decoder token ids -> text."""
    if isinstance(token_ids, list) or (hasattr(token_ids, "shape") and len(token_ids.shape) == 1):
        return decoder_tokenizer.decode(token_ids, skip_special_tokens=True)
    return decoder_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
