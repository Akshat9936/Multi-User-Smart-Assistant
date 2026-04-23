# separator.py
from asteroid.models import ConvTasNet
import torch

class AudioSeparator:
    def __init__(self, model_path="mpariente/ConvTasNet_WHAM!_sepclean"):
        # Load pre-trained model for 2-speaker separation 
        self.model = ConvTasNet.from_pretrained(model_path)
        self.model.eval()

    def separate(self, mixture_tensor):
        # mixture_tensor should be shape (batch, time)
        with torch.no_grad():
            separated = self.model(mixture_tensor)
        return separated # Returns tensor of shape (batch, n_src, time)