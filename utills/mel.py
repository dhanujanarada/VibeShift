import torch
import torchaudio
import torchaudio.transforms as T

def save_mel_with_metadata(audio_path, output_path, genre_id, bpm=None):
    """Save mel spectrogram with metadata as a dictionary."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert stereo to mono BEFORE processing
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 24000 Hz for Vocos compatibility
    if sr != 24000:
        resampler = T.Resample(sr, 24000)
        waveform = resampler(waveform)
        sr = 24000
    
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        f_min=0,
        f_max=8000
    )
    
    mel_spec = mel_transform(waveform)  # Now always (1, 100, time)
    
    # Save as dictionary
    data = {
        'mel': mel_spec,
        'genre_id': genre_id,
        'sample_rate': sr,
        'bpm': bpm,
        'duration': waveform.shape[1] / sr,
        'mel_shape': mel_spec.shape
    }
    
    torch.save(data, output_path)
    print(f"Saved {output_path}: genre={genre_id}, shape={mel_spec.shape}")