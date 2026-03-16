# vocalseperator.py
# Simple audio player for filtered rock dataset

import numpy as np
from datasets import load_from_disk
from pathlib import Path
import sounddevice as sd
import yaml

def load_rock_dataset(dataset_path="data/rock_dataset"):
    """Load filtered rock dataset"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Run raw_dataloader.py first."
        )
    return load_from_disk(str(path))

def play_song(dataset, index):
    """Play a song by index"""
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of range [0, {len(dataset)-1}]")
    
    song = dataset[index]
    audio_array = np.array(song['audio']['array'])
    sample_rate = song['audio']['sampling_rate']
    
    # Display info
    print(f"\nPlaying song {index}:")
    print(f"  Title:  {song.get('title', 'Unknown')}")
    print(f"  Artist: {song.get('artist', 'Unknown')}")
    print(f"  Duration: {len(audio_array)/sample_rate:.2f}s")
    
    # Play audio
    sd.play(audio_array, sample_rate)
    sd.wait()
    print("✓ Finished\n")



# Load dataset
dataset = load_rock_dataset()
print(f"Loaded {len(dataset)} rock songs\n")

# Example usage
if __name__ == "__main__":
    # Play first song
    play_song(dataset, 34)
    
    # Play another song
    # play_song(dataset, 5)
    
    # Play random song
    # import random
    # play_song(dataset, random.randint(0, len(dataset)-1))

    # Load non-rock dataset
    non_rock_dataset, non_rock_genres = load_non_rock_dataset()
    print(f"Loaded {len(non_rock_dataset)} non-rock songs with genres: {non_rock_genres}\n")