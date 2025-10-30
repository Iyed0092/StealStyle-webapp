"""
HybridTextModel combines:
 - EncoderWrapper (RoBERTa)
 - DecoderWrapper (mBART)
 - GAN (Generator + Discriminator)
Provides a research-friendly forward for training and evaluation.
"""

import torch
import torch.nn as nn
from .transformer import EncoderWrapper, DecoderWrapper
from .gan import Generator, Discriminator

class HybridTextModel(nn.Module):
    def __init__(self, device="cpu", embedding_dim=768):
        super().__init__()
        self.device = device
        
        self.encoder = EncoderWrapper(device=device)
        self.decoder = DecoderWrapper(device=device, embedding_dim=embedding_dim)
        # GAN
        self.generator = Generator(embedding_dim)
        self.discriminator = Discriminator(embedding_dim)

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask, labels=None, train_gan=False):

        
        _, sent_repr = self.encoder(enc_input_ids, enc_attention_mask)  

        
        refined = None
        gan_loss = None
        if train_gan:
            refined = self.generator(sent_repr)  
        else:
            refined = sent_repr.detach()

        # Decoder forward with embedding conditioning
        lm_logits = self.decoder.forward_with_embedding(
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=dec_attention_mask,
            encoder_outputs_embeds=refined
        )

        return {
            "lm_logits": lm_logits,
            "refined": refined
        }
