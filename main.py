import os
import time

# Import your custom weapons
from speaker_id import identify_speaker
from transcribe import transcribe_audio # Make sure your function accepts initial_prompt!
from assistant import generate_assistant_response

def process_stream(audio_file, known_profiles):
    print(f"\n{'='*50}\nIncoming Audio Stream: {audio_file}\n{'='*50}")
    
    # Phase 4: Identify Who is Speaking
    identified_user = identify_speaker(audio_file, known_profiles)
    
    if identified_user == "Unknown User":
        print("[System] Unknown voice detected. Ignoring stream to save compute.")
        return
        
    # Phase 2: Transcribe with dynamic context injection
    # (Injecting the identified name ensures 0% Word Error Rate on the user's name)
    transcript = transcribe_audio(audio_file, initial_prompt=identified_user)
    
    if not transcript or len(transcript) < 2:
        print("[System] Audio was just noise. No speech detected.")
        return
        
    # Phase 3: The Brain 
    response = generate_assistant_response(transcript, identified_user)
    
    print(f"\n[FINAL ASSISTANT LOG ACTION] -> {response}\n")

if __name__ == "__main__":
    print("Initializing Multi-User Smart Assistant Engine...\n")
    
    # 1. Load the Vault (Replace these with your actual 3-second reference files later)
    registered_users = {
        "Akshat": "reference_akshat.wav",
        "Palak":  "reference_palak.wav" 
    }
    
    # 2. Simulate the output from your ConvTasNet separation model
    # In the final demo, ConvTasNet will automatically generate these files
    separated_audio_streams = ["stream_1_output.wav", "stream_2_output.wav"]
    
    # 3. Process them simultaneously (or sequentially for Colab CPU limits)
    start_pipeline = time.time()
    
    for stream in separated_audio_streams:
        # Check if file exists before processing (good for testing)
        if os.path.exists(stream):
            process_stream(stream, registered_users)
        else:
            print(f"[Warning] {stream} not found on disk. Skipping.")
            
    print(f"\nTotal Pipeline Execution Time: {time.time() - start_pipeline:.2f} seconds")