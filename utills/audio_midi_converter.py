from ast import main
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import soundfile as sf
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
import shutil
import subprocess
import time
from pydub import AudioSegment


class AudioMidiConverter:
    """Convert audio to MIDI and MIDI back to audio using basic_pitch and FluidSynth."""
    
    def __init__(self, soundfont_path: Optional[str] = None):
        """
        Initialize the converter.
        
        Args:
            soundfont_path: Path to soundfont file. If None, uses default TimGM6mb.sf2
        """
        if soundfont_path is None:
            script_dir = Path(__file__).parent.parent
            soundfont_path = str(script_dir / "data" / "TimGM6mb.sf2")
        
        self.soundfont_path = os.path.abspath(soundfont_path)
        
        if not os.path.exists(self.soundfont_path):
            raise FileNotFoundError(f"Soundfont file not found: {self.soundfont_path}")
        
        self.fluidsynth_path = self._find_fluidsynth()
    
    def _find_fluidsynth(self) -> str:
        """Find FluidSynth executable on the system."""
        fluidsynth_path = shutil.which("fluidsynth")
        if not fluidsynth_path:
            possible_paths = [
                r"C:\Program Files\FluidSynth\bin\fluidsynth.exe",
                r"C:\Program Files (x86)\FluidSynth\bin\fluidsynth.exe",
                os.path.expanduser(r"~\scoop\apps\fluidsynth\current\bin\fluidsynth.exe"),
                os.path.expanduser(r"~\scoop\shims\fluidsynth.exe"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    fluidsynth_path = path
                    break
            
            if not fluidsynth_path:
                raise FileNotFoundError(
                    "FluidSynth not found. Please install it:\n"
                    "  Windows: scoop install fluidsynth\n"
                    "  Or download from: https://github.com/FluidSynth/fluidsynth/releases"
                )
        
        return fluidsynth_path
    
    def audio_to_midi(self, audio_path: str, output_dir: str) -> str:
        """
        Convert audio file to MIDI using basic_pitch.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save MIDI file
            
        Returns:
            Path to generated MIDI file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Predicting MIDI for {audio_path}...")
        model_output, midi_data, note_events = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        input_path = Path(audio_path)
        base_name = input_path.stem
        midi_output_path = os.path.join(output_dir, f"{base_name}_basic_pitch.mid")
        
        midi_data.write(midi_output_path)
        print(f"MIDI file created: {midi_output_path}")
        
        return midi_output_path
    
    def midi_to_audio(self, midi_path: str, output_path: str) -> str:
        
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        midi_path = os.path.abspath(midi_path)
        output_path = os.path.abspath(output_path)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # FluidSynth creates WAV first, then convert to MP3 if needed
        temp_wav = output_path.replace('.mp3', '_temp.wav') if output_path.endswith('.mp3') else output_path
        
        cmd = [
            self.fluidsynth_path,
            '-ni',
            '-F', temp_wav,
            '-r', '44100',
            self.soundfont_path,
            midi_path
        ]
        
        print(f"Converting MIDI to audio...")
        print(f"  FluidSynth: {self.fluidsynth_path}")
        print(f"  Input MIDI: {midi_path}")
        print(f"  Output: {output_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FluidSynth failed with return code {result.returncode}:\n{result.stderr}")
        
        time.sleep(1)
        
        if not os.path.exists(temp_wav):
            raise RuntimeError(f"FluidSynth did not create output file: {temp_wav}")
        
        # Convert WAV to MP3 if output is MP3
        if output_path.endswith('.mp3'):
            print(f"Converting WAV to MP3...")
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format='mp3', bitrate='192k')
            os.remove(temp_wav)  # Remove temporary WAV file
            file_size = os.path.getsize(output_path)
            print(f"MP3 file created: {output_path} ({file_size} bytes)")
        else:
            file_size = os.path.getsize(output_path)
            print(f"Audio file created: {output_path} ({file_size} bytes)")
        
        return output_path
    
    def convert(self, audio_path: str, output_dir: str) -> str:
        """
        Convert audio to MIDI and back to audio.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save output files
            
        Returns:
            Path to resynthesized audio file
        """
        midi_path = self.audio_to_midi(audio_path, output_dir)
        
        input_path = Path(audio_path)
        output_audio = os.path.join(output_dir, f"{input_path.stem}_synth.mp3")
        
        audio_path = self.midi_to_audio(midi_path, output_audio)
        return audio_path









