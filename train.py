import torch
import torchaudio
import random
from torch.utils.data import DataLoader
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import ConvTasNet
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Opening LibriSpeech Data Stream... (0GB Disk Space Required)")
librispeech = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.100",
    streaming=True
)

class StreamingMixDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_streaming_dataset, sample_rate=8000, duration=3):
        self.dataset = hf_streaming_dataset
        self.sr = sample_rate
        self.length = duration * sample_rate
        self._buffer = []

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        
        while True:
            while len(self._buffer) < 100:
                try:
                    item = next(dataset_iter)
                    audio = torch.tensor(item['audio']['array'], dtype=torch.float32)
                    if item['audio']['sampling_rate'] != self.sr:
                        audio = torchaudio.functional.resample(audio, item['audio']['sampling_rate'], self.sr)
                    for i in range(0, len(audio) - self.length, self.length):
                        self._buffer.append(audio[i:i+self.length])
                except StopIteration:
                    break
                    
            if len(self._buffer) < 2:
                break
                
            idx1, idx2 = random.sample(range(len(self._buffer)), 2)
            s1, s2 = self._buffer[idx1], self._buffer[idx2]
            self._buffer.pop(0)

            snr_db = random.uniform(-5, 5)
            scale = 10 ** (snr_db / 20)
            mix = s1 + scale * s2

            # --- DOMAIN GENERALIZATION (DIRTY DATA) ---
            if random.random() > 0.3:
                rir_length = int(0.3 * self.sr)
                decay = torch.exp(-torch.linspace(0, 6, rir_length))
                rir = decay * torch.randn(rir_length)
                rir = rir / rir.norm()
                mix_fft = torch.fft.rfft(mix, n=mix.shape[-1]+rir_length-1)
                rir_fft = torch.fft.rfft(rir, n=mix.shape[-1]+rir_length-1)
                mix = torch.fft.irfft(mix_fft * rir_fft)[:mix.shape[-1]]

            noise_snr = random.uniform(10, 25)
            noise = torch.randn_like(mix)
            noise_scale = mix.std() / (noise.std() * (10 ** (noise_snr / 20)) + 1e-8)
            mix = mix + noise_scale * noise
            # ------------------------------------------

            mix = mix / (mix.abs().max() + 1e-8)
            s1  = s1  / (s1.abs().max()  + 1e-8)
            s2  = s2  / (s2.abs().max()  + 1e-8)

            sources = torch.stack([s1, scale * s2])
            yield {'mix': mix.unsqueeze(0), 'sources': sources}

# 1. Setup Data
stream_dataset = StreamingMixDataset(librispeech)
train_loader = DataLoader(stream_dataset, batch_size=2) 

# 2. Load Architecture (MAX CAPACITY: ~4.9M Parameters)
model = ConvTasNet(
    n_src=2, 
    n_repeats=3,  # Doubled capacity
    n_blocks=8,   # Doubled capacity
    n_filters=256, 
    sample_rate=8000
)

# NO LOAD_STATE_DICT. We are starting with a fresh brain.
print("Initializing fresh 4.9M parameter weights...")

model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') 
scaler = torch.amp.GradScaler('cuda')

print("Engine started. Training on Augmented/Noisy LibriSpeech Stream...")

# 3. Training Loop
EPOCHS = 20
STEPS_PER_EPOCH = 200 
best_loss = float('inf')

train_iter = iter(train_loader)

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        mix = batch['mix'].to(device) 
        targets = batch['sources'].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            preds = model(mix)
            loss = loss_func(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    print(f"--- Epoch {epoch+1}/{EPOCHS} Complete | Avg Loss: {avg_loss:.4f} ---")
    
    if avg_loss < best_loss:
        print(f"[*] New best loss achieved! Saving checkpoint...")
        best_loss = avg_loss
        torch.save(model.state_dict(), "/content/Multi-User-Smart-Assistant/checkpoint_best_model.pt")

print("Training successfully finished!")