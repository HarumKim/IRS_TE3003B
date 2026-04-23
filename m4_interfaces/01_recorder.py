import sounddevice as sd
from scipy.io import wavfile
import numpy as np
import os
import time

SAMPLE_RATE = 16000  # 16 kHz required
DURATION = 2.0       # Seconds per recording
REPETITIONS = 15     # Times per word
ROOT_FOLDER = "dataset_voz"

# List of words
words = [
    "start", "stop", "left", "right", "forward", 
    "back", "lift", "lower", "fast", "slow"
]

def record_audio(file_name):
    print(f"  -> Recording... SPEAK NOW!")
    # Record audio
    recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, dtype='int16')
    sd.wait()  # Wait for the time to finish
    wavfile.write(file_name, SAMPLE_RATE, recording)
    print(f"  -> Saved to: {file_name}")

# Create main folder
if not os.path.exists(ROOT_FOLDER):
    os.makedirs(ROOT_FOLDER)

print("--- START OF DATA COLLECTION ---")
print(f"Instructions: You will say each word {REPETITIONS} times.")

for word in words:
    # Create subfolder for the word
    word_path = os.path.join(ROOT_FOLDER, word)
    if not os.path.exists(word_path):
        os.makedirs(word_path)
    
    print(f"\n>>> PREPARING WORD: [{word.upper()}]")
    
    for i in range(1, REPETITIONS + 1):
        input(f"Press ENTER to record repetition {i}/{REPETITIONS}...")
        
        # File name: dataset_voz/start/start_1.wav
        file_path = os.path.join(word_path, f"{word}_{i}.wav")
        
        record_audio(file_path)
        time.sleep(0.5) # Brief pause to avoid saturation

print("\n--- PRACTICE COMPLETED! ---")
print(f"All files are in the '{ROOT_FOLDER}' folder")
