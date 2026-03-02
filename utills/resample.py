import torch
import torchaudio
from pathlib import Path

def resample_dataset(input_dir, output_dir, target_sr=44100):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    audio_files = list(Path(input_dir).glob('**/*.wav')) + \
                  list(Path(input_dir).glob('**/*.mp3')) + \
                  list(Path(input_dir).glob('**/*.flac'))
    
    print(f"Found {len(audio_files)} audio files in {input_dir}", flush=True)
    if not audio_files:
        print("No audio files found. Exiting.", flush=True)
        return
    
    for i, file_path in enumerate(audio_files, 1):
        try:
            print(f"[{i}/{len(audio_files)}] Processing: {file_path}", flush=True)
            waveform, sr = torchaudio.load(file_path)
            
            print(f"    Sample rate: {sr} Hz", end="", flush=True)
            
            if sr != target_sr:
                # High-quality resampling (equivalent to librosa kaiser_best)
                resampler = torchaudio.transforms.Resample(
                    sr, 
                    target_sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492
                )
                waveform = resampler(waveform)
                print(f" → {target_sr} Hz", flush=True)
            else:
                print(" (already 44.1kHz)", flush=True)
            
            output_path = Path(output_dir) / file_path.name
            torchaudio.save(str(output_path), waveform, target_sr)
            print(f"    Saved: {output_path}", flush=True)
        except Exception as exc:
            print(f"    Error processing {file_path}: {exc}", flush=True)
            continue

def preprocess_mono(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_files = (
        list(Path(input_dir).glob('**/*.wav')) +
        list(Path(input_dir).glob('**/*.mp3')) +
        list(Path(input_dir).glob('**/*.flac'))
    )

    for file_path in audio_files:
        waveform, sr = torchaudio.load(file_path)
        original_channels = waveform.shape[0]
        if original_channels > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        new_channels = waveform.shape[0]
        print(f"Processing: {file_path.name} | Channels: {original_channels} → {new_channels}", flush=True)
        output_path = Path(output_dir) / file_path.name
        torchaudio.save(str(output_path), waveform, sr)


def main():
    target_sr = 44100

    input = r"C:\Users\Dhanuja\Downloads\dataset\instrumentals"
    output = r"C:\Users\Dhanuja\Downloads\dataset\preprocessed audio"
    
    resample_dataset(input, output, target_sr)
    preprocess_mono(input_dir=output, output_dir=output)

if __name__ == "__main__":
    main()