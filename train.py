# ==========================================
# HOTFIX: Monkey-patch random.randint to fix Asteroid + Python 3.12 bug
import random
_original_randint = random.randint
def _patched_randint(a, b):
    return _original_randint(int(a), int(b))
random.randint = _patched_randint
# ==========================================

import torch
from torch.utils.data import DataLoader
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data import LibriMix
from asteroid.models import ConvTasNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Architecture (< 5 Million Parameters)
model = ConvTasNet(n_src=2, n_repeats=2, n_blocks=6, n_filters=256, sample_rate=8000)
model = model.to(device)
model.train()

# Verify KPI for the Hackathon Judges
total_params = sum(p.numel() for p in model.parameters())
print(f"MODEL LOADED! Total Parameters: {total_params:,}")
if total_params < 5000000:
    print("KPI MET: Model is strictly under 5 Million parameters.")

# 2. Setup Data
train_set = LibriMix(csv_dir='MiniLibriMix/metadata/train', task='sep_clean', sample_rate=8000)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

# 3. Setup Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') 
scaler = torch.amp.GradScaler('cuda')

print("Engine started. Training on MiniLibriMix train set...")

# 4. Training Loop
for epoch in range(2):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        
        # Asteroid's ConvTasNet expects the [Batch, 1, Time] shape, so we do NOT squeeze
        mix, targets = batch
        mix = mix.to(device) 
        targets = targets.to(device)

        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda'):
            # ConvTasNet automatically handles the encoder/mask/decoder internally!
            preds = model(mix)
            loss = loss_func(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Clip gradients using the correct 'model.parameters()'
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
    
    # Save the updated model dictionary
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")
    
print("Training successfully finished!")