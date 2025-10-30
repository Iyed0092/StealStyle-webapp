# backend/hybrid_text/train.py
"""
Robust training script for the HybridTextModel (Transformer + embedding-level GAN).
Includes: safe GAN update order, debug prints, grad clipping, checkpointing,
and a dry-run mode to validate a single-batch forward/backward pass.

Drop this file into hybrid_text/ and run:
    python -m hybrid_text.train
or import train_loop(...) into a notebook.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports (your package)
from hybrid_text.data.load_data import ProseDataset
from hybrid_text.models.hybrid_model import HybridTextModel
from hybrid_text.utils import set_seed, save_checkpoint

# ----------------------------
# Config / Hyperparameters
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
LR_TRANS = 3e-5
LR_GAN = 1e-4
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = "checkpoints"
DRY_RUN_ONE_BATCH = False   # set True to run only a single batch (sanity check)
LOG_EVERY = 50

# ----------------------------
# Utility: safe training loop
# ----------------------------
def train_loop(
    out_dir=CHECKPOINT_DIR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr_trans=LR_TRANS,
    lr_gan=LR_GAN,
    max_grad_norm=MAX_GRAD_NORM,
    device=DEVICE,
    dry_run_one_batch=DRY_RUN_ONE_BATCH
):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset (uses your ProseDataset wrapper)
    train_ds = ProseDataset(split="train", max_length=128, limit=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: ProseDataset.collate_fn(b, max_length=128))

    # Instantiate model and move to device
    model = HybridTextModel(device=device).to(device)
    model.train()

    # Parameter groups: transformer (encoder+decoder+proj/embedding pipeline) vs GAN (generator+discriminator)
    # Adjust these lists if your HybridTextModel structure differs.
    params_trans = list(model.encoder.model.parameters()) + list(model.decoder.model.parameters()) + list(getattr(model.decoder, "proj", nn.Module()).parameters())
    params_gan = list(model.generator.parameters()) + list(model.discriminator.parameters())

    optimizer_trans = torch.optim.Adam(params_trans, lr=lr_trans)
    optimizer_gan = torch.optim.Adam(params_gan, lr=lr_gan)

    # Losses
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)  # labels use -100
    bce_loss = nn.BCEWithLogitsLoss()

    # Safety: detect autograd anomalies while debugging
    torch.autograd.set_detect_anomaly(True)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_gan_accum = 0.0

        for step, batch in enumerate(train_loader):
            global_step += 1

            # Move batch to device
            enc_ids = batch["enc_input_ids"].to(device)
            enc_mask = batch["enc_attention_mask"].to(device)
            dec_ids = batch["dec_input_ids"].to(device)
            dec_mask = batch["dec_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward (Transformer + generator path inside HybridTextModel)
            out = model(enc_ids, enc_mask, dec_ids, dec_mask, labels=labels, train_gan=True)
            lm_logits = out["lm_logits"]   # (B, seq_len+1, vocab)
            refined = out["refined"]       # (B, d_model)


            with torch.no_grad():
                _, real_repr = model.encoder(enc_ids, enc_mask)


            # --- Transformer CE loss ---
            logits_trim = lm_logits[:, :labels.size(1), :].contiguous()  # align shapes
            loss_ce = ce_loss(logits_trim.view(-1, logits_trim.size(-1)), labels.view(-1))


            optimizer_gan.zero_grad()
            fake_logits_for_d = model.discriminator(refined.detach())
            real_logits_for_d = model.discriminator(real_repr.detach())

            loss_d = 0.5 * (bce_loss(real_logits_for_d, torch.ones_like(real_logits_for_d)) +
                            bce_loss(fake_logits_for_d, torch.zeros_like(fake_logits_for_d)))
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_grad_norm)
            optimizer_gan.step()


            for p in model.discriminator.parameters():
                p.requires_grad = False

            optimizer_trans.zero_grad()
            fake_logits_for_g = model.discriminator(refined)
            loss_g = bce_loss(fake_logits_for_g, torch.ones_like(fake_logits_for_g))

            total_loss = loss_ce + 0.1 * loss_g   
            total_loss.backward()

            # Clip grads on transformer/generator param group
            torch.nn.utils.clip_grad_norm_(params_trans, max_grad_norm)
            optimizer_trans.step()

            # unfreeze discriminator
            for p in model.discriminator.parameters():
                p.requires_grad = True

            # === Logging / accumulation ===
            epoch_ce += loss_ce.item()
            epoch_gan_accum += 0.5 * (loss_d.item() + loss_g.item())

            if step % LOG_EVERY == 0:
                print(f"[Epoch {epoch} Step {step}] CE={loss_ce.item():.4f} D={loss_d.item():.4f} G={loss_g.item():.4f}")

            if dry_run_one_batch:
                print("Dry-run enabled — stopping after 1 batch.")
                # Save a checkpoint snapshot if helpful
                ckpt_path = os.path.join(out_dir, f"hybrid_epoch{epoch}_dryrun.pt")
                save_checkpoint(model, optimizer_trans, ckpt_path, epoch, meta={"step": step})
                return model

        avg_ce = epoch_ce / max(1, len(train_loader))
        avg_gan = epoch_gan_accum / max(1, len(train_loader))
        print(f"Epoch {epoch} finished. Avg CE: {avg_ce:.4f} Avg GAN-term: {avg_gan:.4f}")

        # Save checkpoint at epoch end
        ckpt_path = os.path.join(out_dir, f"hybrid_epoch{epoch}.pt")
        save_checkpoint(model, optimizer_trans, ckpt_path, epoch, meta={"avg_ce": avg_ce, "avg_gan": avg_gan})

    return model


if __name__ == "__main__":
    # quick safety: one-batch dry-run first to confirm everything's working
    train_loop(dry_run_one_batch=True)   # switch to False to run full training
