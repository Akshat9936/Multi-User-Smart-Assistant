# main.py
from separator import AudioSeparator
from kws_model import KeywordSpotter
import time

# Initialize modules
separator = AudioSeparator()
kws = KeywordSpotter()

def run_pipeline(mixture):
    start_time = time.time()
    
    # 1. Separation
    separated_audio = separator.separate(mixture)
    
    # 2. Parallel Keyword Detection
    for i, source in enumerate(separated_audio):
        if kws.predict(source):
            print(f"Keyword detected in Source {i}!")
            
    # Calculate xRT 
    duration = time.time() - start_time
    print(f"Pipeline xRT: {duration:.4f}")

# Example execution
# run_pipeline(mixture_data)