import torch
import torchaudio
import os
import subprocess
import time
from asteroid.models import ConvTasNet

def separate_and_process(mixed_audio_path, weights_path):
    print(f"{'='*50}\nPHASE 1: ACOUSTIC SEPARATION\n{'='*50}")
    print(f"Loading custom <5M Parameter ConvTasNet weights from [{weights_path}]...")
    
    start_time = time.time()
    
    try:
        model = ConvTasNet.from_pretrained(weights_path)
    except Exception as e:
        print(f"[Fallback] Direct state_dict load initiated...")
        # THE FIX: Matching the exact architecture from train.py
        model = ConvTasNet(n_src=2, n_repeats=3, n_blocks=8, n_filters=256, sample_rate=8000)
        
        # Load the weights into the correctly shaped shell
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    model.eval()
    
    print(f"Listening to chaotic audio: {mixed_audio_path}...")
    mix, sample_rate = torchaudio.load(mixed_audio_path)
    
    # Resample to 8000Hz if necessary (since your model was trained on 8000Hz)
    if sample_rate != 8000:
        print("Resampling audio to 8000Hz to match model constraints...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        mix = resampler(mix)
        sample_rate = 8000

    if mix.shape[0] > 1:
        mix = mix.mean(dim=0, keepdim=True)
        
    print("Untangling overlapping voices...")
    with torch.no_grad():
        separated_sources = model(mix)
        
    stream_1 = separated_sources[0, 0, :].unsqueeze(0)
    stream_2 = separated_sources[0, 1, :].unsqueeze(0)
    
    torchaudio.save("stream_1_output.wav", stream_1, sample_rate)
    torchaudio.save("stream_2_output.wav", stream_2, sample_rate)
    
    separation_time = time.time() - start_time
    print(f"[Success] Audio separated into two clean streams in {separation_time:.2f} seconds!")
    
    print("\nTriggering Smart Assistant Pipeline...")
    subprocess.run(["python", "main.py"])

if __name__ == "__main__":
    WEIGHTS_FILE = "/content/checkpoint_best_model.pt"
    TEST_AUDIO = "/content/noisy_room_test.wav" 
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"CRITICAL ERROR: {WEIGHTS_FILE} not found in directory.")
    elif not os.path.exists(TEST_AUDIO):
        print(f"WAITING: Please create or upload a test file named '{TEST_AUDIO}'.")
    else:
        separate_and_process(TEST_AUDIO, WEIGHTS_FILE)