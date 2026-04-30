import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import time

def identify_speaker(unknown_audio_path, profiles_dict):
    print("Loading ECAPA-TDNN Speaker Verification Model...")
    # This model is tiny and fast, perfectly compliant with your KPIs
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    print(f"Analyzing voice signature from {unknown_audio_path}...")
    start_time = time.time()
    
    best_match = "Unknown User"
    highest_score = 0.0
    
    # We compare the unknown voice against known user profile audio clips
    for user_name, reference_audio in profiles_dict.items():
        # The model returns a similarity score and a boolean prediction
        score, prediction = verification.verify_files(unknown_audio_path, reference_audio)
        
        # Convert tensor to float
        score_val = score.item()
        
        if score_val > highest_score and prediction.item() == True:
            highest_score = score_val
            best_match = user_name

    end_time = time.time()
    
    print("\n--- Speaker ID Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    if best_match == "Unknown User":
        print("Result: No match found. Background noise or unregistered user.")
    else:
        print(f"Result: Voice matched to [{best_match}] with confidence {highest_score:.2f}")
        
    return best_match

if __name__ == "__main__":
    print("Speaker Verification Module Ready.")
    
    # --- HOW IT WORKS IN THE FINAL PIPELINE ---
    # 1. You would have a 3-second reference clip of you saying anything.
    # 2. You would have a 3-second reference clip of another user.
    # 
    # known_users = {
    #     "Akshat": "akshat_reference_voice.wav",
    #     "Palak": "palak_reference_voice.wav" 
    # }
    # 
    # identified_name = identify_speaker("separated_audio_stream_1.wav", known_users)
    # 
    # 3. THEN you pass 'identified_name' dynamically into Whisper's initial_prompt!