
import torch
from hybrid_text.models.hybrid_model import HybridTextModel
from hybrid_text.data.preprocess import detokenize_decoder, encode_encoder
from hybrid_text.utils import set_seed, load_checkpoint

class InferEngine:
    def __init__(self, checkpoint_path=None, device=None):
        set_seed(42)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = HybridTextModel(device=self.device).to(self.device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"[INFO] Loaded checkpoint {checkpoint_path}")
        self.model.eval()

    def generate(self, texts, max_length=120, num_beams=4):
        enc = encode_encoder(texts, max_length=max_length)
        enc_ids = enc["input_ids"].to(self.device)
        enc_mask = enc["attention_mask"].to(self.device)
        gen_ids = self.model.generate(enc_ids, enc_mask, max_length=max_length, num_beams=num_beams)
        return detokenize_decoder(gen_ids)
