import google.generativeai as genai
import os

# Set your API key here (get one free at aistudio.google.com if you haven't yet)
# In a real production app, use environment variables. For a hackathon demo, this is fine.
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def generate_assistant_response(transcribed_text, speaker_id):
    print(f"Analyzing intent for {speaker_id}...")
    
    # We use gemini-1.5-flash for maximum speed to help with the xRT KPI
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # The System Prompt: Strict instructions for the Smart Home logic
    system_context = f"""
    You are a highly efficient Smart Home Assistant processing a voice command in a chaotic environment.
    The identified speaker is: {speaker_id}.
    
    Your task:
    1. Determine if the text is a legitimate command or just background noise/chatter.
    2. If it is chatter, respond with exactly: [NO ACTION REQUIRED]
    3. If it is a command, provide a brief, personalized confirmation of the action you are taking, explicitly addressing the speaker by their tag ({speaker_id}).
    Keep your response strictly under 2 sentences to ensure real-time audio playback latency is low.
    """
    
    prompt = f"{system_context}\n\nUser Command: '{transcribed_text}'\nAssistant Response:"
    
    # Generate the response
    response = model.generate_content(prompt)
    
    return response.text.strip()

if __name__ == "__main__":
    # --- Quick Local Test ---
    print("Testing Assistant Brain...")
    test_text = "Turn off the living room lights, I'm going to sleep."
    test_speaker = "Speaker 1"
    
    reply = generate_assistant_response(test_text, test_speaker)
    print(f"\n[Final Output] -> {reply}")