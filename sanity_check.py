# sanity_check.py
from dataset import SmartAssistantDataset
from torch.utils.data import DataLoader

# UPDATE THESE PATHS to match your Google Drive folders
# Example: '/content/drive/MyDrive/LibriMix/wav16k/min/train-360/mix_clean'
MIXTURE_PATH = "/content/drive/MyDrive/path_to_your_mixture_folder"
SOURCE_PATH = "/content/drive/MyDrive/path_to_your_source_folder"

dataset = SmartAssistantDataset(mixture_dir=MIXTURE_PATH, source_dir=SOURCE_PATH)
loader = DataLoader(dataset, batch_size=1)

try:
    mix, src = next(iter(loader))
    print(f"SUCCESS! Mixture shape: {mix.shape}, Source shape: {src.shape}")
except Exception as e:
    print(f"Error loading data: {e}")