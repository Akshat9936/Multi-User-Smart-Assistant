import os
import time
from transcribe import transcribe_audio 
from assistant import generate_assistant_response

def process_stream(audio_file, speaker_tag):
    print(f"\n{'='*50}\nProcessing {speaker_tag} ({audio_file})\n{'='*50}")
    
    # Phase 2: Transcribe (No hardcoded names, just raw speed)
    transcript = transcribe_audio(audio_file)
    
    if not transcript or len(transcript) < 2:
        print(f"[System] {speaker_tag} stream was just noise. Ignored.")
        return
        
    # Phase 3: The Brain (Personalizes based on the assigned speaker_tag)
    response = generate_assistant_response(transcript, speaker_tag)
    
    print(f"\n[FINAL ASSISTANT ACTION] -> {response}\n")

if __name__ == "__main__":
    print("Initializing Multi-User Smart Assistant Engine...\n")
    
    # Simulate the output files from your ConvTasNet separation model
    # (In the final demo, your separator will generate these in real-time)
    separated_audio_streams = {
        "Speaker 1": "stream_1_output.wav", 
        "Speaker 2": "stream_2_output.wav"
    }
    
    start_pipeline = time.time()
    
    # Process each separated speaker
    for speaker_tag, stream_file in separated_audio_streams.items():
        if os.path.exists(stream_file):
            process_stream(stream_file, speaker_tag)
        else:
            print(f"[Warning] {stream_file} not found on disk. Skipping.")
            
    print(f"\nTotal End-to-End Pipeline Time: {time.time() - start_pipeline:.2f} seconds")