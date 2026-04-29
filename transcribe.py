import whisper
import torch
import time
import warnings

# Suppress standard Whisper warnings for cleaner terminal output
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    
    # Auto-detect hardware: uses GPU if available, falls back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the lightweight 'base' model to meet the < 0.5 xRT Hackathon KPI
    model = whisper.load_model("base", device=device)
    
    print(f"Listening to {audio_path}...")
    start_time = time.time()
    
    # Dynamically disable fp16 if on CPU to prevent crashes
    use_fp16 = True if device == "cuda" else False
    result = model.transcribe(audio_path, fp16=use_fp16)
    
    end_time = time.time()
    text_output = result['text'].strip()
    
    print("\n--- Transcription Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Recognized Text: '{text_output}'")
    
    return text_output

if __name__ == "__main__":
    # This block allows you to test the file directly, but keeps the 
    # function clean when imported into your final Phase 4 pipeline.
    print("ASR Module Ready. Import 'transcribe_audio' to use in the main pipeline.")