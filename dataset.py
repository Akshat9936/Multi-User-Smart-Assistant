import torch
from torch.utils.data import Dataset
import torchaudio
import os

class SmartAssistantDataset(Dataset):
    def __init__(self, mixture_dir, source_dir):
        """
        mixture_dir: Path to folders containing mixed audio files 
        source_dir: Path to folders containing individual speaker files 
        """
        self.mixture_dir = mixture_dir
        self.source_dir = source_dir
        self.files = os.listdir(mixture_dir) # List of all mixture files 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Load the mixture (the noisy input)
        mix_path = os.path.join(self.mixture_dir, self.files[idx])
        mixture, sr = torchaudio.load(mix_path)
        
        # 2. Load the ground truth sources (Speaker 1, Speaker 2)
        # Note: Adjust these paths based on your specific dataset folder structure
        s1, _ = torchaudio.load(os.path.join(self.source_dir, "s1", self.files[idx]))
        s2, _ = torchaudio.load(os.path.join(self.source_dir, "s2", self.files[idx]))
        
        # 3. Combine sources into a single tensor for the model
        sources = torch.stack([s1, s2], dim=0)
        
        return mixture, sources