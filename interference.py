import torch
import torchaudio
import os
import subprocess
import time
from asteroid.models import ConvTasNet

def separate_and_process(mixed_audio_path, weights_path):
    print(f"{'='*50}\nPHASE 1: ACOUSTIC SEPARATION\n{'='*50}")
    print(f"Loading custom ConvTasNet weights from [{weights_path}]...")
    
    start_time = time.time()
    
    # 1. Load your trained Hackathon model
    # (If your train.py used standard PyTorch saving, we load it here)
    try:
        model = ConvTasNet.from_pretrained(weights_path)
    except Exception as e:
        print(f"[Warning] Standard load failed, attempting direct state_dict load... Error: {e}")
        # Fallback for basic PyTorch state_dicts
        model = ConvTasNet(n_src=2) # Configured for 2 speakers
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    model.eval()
    
    # 2. Load the chaotic room audio
    print(f"Listening to chaotic audio: {mixed_audio_path}...")
    mix, sample_rate = torchaudio.load(mixed_audio_path)
    
    # Force mono-channel for the model
    if mix.shape[0] > 1:
        mix = mix.mean(dim=0, keepdim=True)
        
    # 3. Execute the Separation
    print("Untangling overlapping voices...")
    with torch.no_grad():
        # The model returns a tensor with the separated sources
        separated_sources = model(mix)
        
    # 4. Save the separated streams for the Assistant
    # Shape is typically (batch, n_sources, time) -> (1, 2, time)
    stream_1 = separated_sources[0, 0, :].unsqueeze(0)
    stream_2 = separated_sources[0, 1, :].unsqueeze(0)
    
    torchaudio.save("stream_1_output.wav", stream_1, sample_rate)
    torchaudio.save("stream_2_output.wav", stream_2, sample_rate)
    
    separation_time = time.time() - start_time
    print(f"[Success] Audio separated into two clean streams in {separation_time:.2f} seconds!")
    
    # 5. Hand off to the Smart Assistant (Phase 2 & 3)
    print("\nTriggering Smart Assistant Pipeline...")
    subprocess.run(["python", "main.py"])


if __name__ == "__main__":
    # The name of your newly trained file
    WEIGHTS_FILE = "checkpoint_epoch14.pt"
    
    # A test file containing two people talking at once
    TEST_AUDIO = "noisy_room_test.wav" 
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"CRITICAL ERROR: {WEIGHTS_FILE} not found in directory.")
    elif not os.path.exists(TEST_AUDIO):
        print(f"WAITING: Please upload a test audio file named '{TEST_AUDIO}' containing overlapping speech.")
    else:
        separate_and_process(TEST_AUDIO, WEIGHTS_FILE)