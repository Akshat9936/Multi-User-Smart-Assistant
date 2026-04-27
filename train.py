import torch
import wandb
from asteroid.data import LibriMix
from speechbrain.inference import SepformerSeparation
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# 1. Initialize W&B
wandb.init(project="smart-assistant-hackathon")

# 2. Get the loaders automatically
train_loader, val_loader = LibriMix.loaders_from_mini(task='sep_clean', batch_size=4)

# 3. Load Model
model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained")

# 4. Training Loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    for batch in train_loader:
        # mix is the audio mixture, targets are the separated sources
        mix = batch['mix']
        targets = batch['sources']
        
        optimizer.zero_grad()
        preds = model.separate_batch(mix)
        
        # Loss: Scale-Invariant SNR
        loss = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')(preds, targets)
        loss.backward()
        optimizer.step()
        
        wandb.log({"loss": loss.item()})
        print(f"Loss: {loss.item()}")