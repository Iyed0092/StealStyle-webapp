"""
Encoder (RoBERTa) + Decoder (mBART) wrapper with a projection to inject embedding conditioning.
This file exposes a research-friendly wrapper that:
 - computes encoder sentence embeddings,
 - provides a method to decode conditioned on a provided embedding (prefix injection).
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


ENCODER_MODEL = "roberta-base"
DECODER_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

class EncoderWrapper(nn.Module):
    def __init__(self, model_name=ENCODER_MODEL, device="cpu"):
        super().__init__()
        self.device = device
        self.model = RobertaModel.from_pretrained(model_name).to(device)
        self.model.eval()  

    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  
        mask = attention_mask.unsqueeze(-1)
        masked = last_hidden * mask
        sum_hidden = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        sent_repr = sum_hidden / lengths
        return last_hidden, sent_repr


class DecoderWrapper(nn.Module):
    def __init__(self, model_name=DECODER_MODEL, device="cpu", embedding_dim=768):
        super().__init__()
        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        
        self.proj = nn.Linear(embedding_dim, self.model.config.d_model)

    

    def forward_with_embedding(self, decoder_input_ids, decoder_attention_mask, encoder_outputs_embeds):
        
        prefix = self.proj(encoder_outputs_embeds).unsqueeze(1) 

        
        token_embeds = self.model.model.decoder.embed_tokens(decoder_input_ids)
        conditioned = torch.cat([prefix, token_embeds], dim=1)

        
        if decoder_attention_mask is not None:
            prefix_mask = torch.ones((decoder_attention_mask.size(0), 1), dtype=decoder_attention_mask.dtype, device=decoder_attention_mask.device)
            decoder_attention_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)

      
        encoder_outputs = BaseModelOutput(
            last_hidden_state=prefix, 
            hidden_states=None,
            attentions=None
        )

        # forward pass through MBART
        outputs = self.model.model(
            encoder_outputs=encoder_outputs,  
            decoder_inputs_embeds=conditioned,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True
        )

        # map to vocab
        lm_logits = self.model.lm_head(outputs.last_hidden_state)
        return lm_logits



    def generate_from_embedding(self, embedding, max_length=100, num_beams=4):
        """
        Generate sequences from a projected embedding.
        Note: This prepends a start token and uses MBart generation; advanced conditioning requires custom loops.
        """
        batch_size = embedding.size(0)
        start_token_id = self.model.config.decoder_start_token_id
        input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=embedding.device)
        generated = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        return generated
