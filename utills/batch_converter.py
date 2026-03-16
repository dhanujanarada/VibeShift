"""
Batch converter utility for audio to MIDI and MIDI to audio conversion.
Provides easy-to-use interface for batch processing multiple files.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

from audio_midi_converter import BatchConverter, AudioMidiConverter


def batch_convert_directory(input_dir: str, output_dir: str, 
                           audio_formats: List[str] = None,
                           soundfont_path: Optional[str] = None) -> Dict[str, str]:
    """
    Convert all audio files in a directory to synthesized audio.
    
    Usage:
        python batch_converter.py <input_dir> <output_dir>
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save converted files
        audio_formats: List of audio extensions to process (default: ['.wav', '.mp3', '.flac'])
        soundfont_path: Path to custom soundfont file (optional)
        
    Returns:
        Dictionary mapping input files to output files
    """
    batch_converter = BatchConverter(soundfont_path)
    return batch_converter.convert_directory(input_dir, output_dir, audio_formats)


def batch_convert_files(file_list: List[str], output_dir: str,
                       soundfont_path: Optional[str] = None) -> Dict[str, str]:
    """
    Convert a specific list of audio files.
    
    Args:
        file_list: List of audio file paths
        output_dir: Directory to save converted files
        soundfont_path: Path to custom soundfont file (optional)
        
    Returns:
        Dictionary mapping input files to output files
    """
    batch_converter = BatchConverter(soundfont_path)
    return batch_converter.convert_files(file_list, output_dir)


def main():
    """Command-line interface for batch conversion."""
    if len(sys.argv) < 3:
        print("Batch Audio Converter - Convert all audio files in a directory")
        print("\nUsage:")
        print("  python batch_converter.py <input_dir> <output_dir> [soundfont_path]")
        print("\nExamples:")
        print("  python batch_converter.py ./input ./output")
        print("  python batch_converter.py ./input ./output ./custom_soundfont.sf2")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    soundfont_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    results = batch_convert_directory(input_dir, output_dir, soundfont_path=soundfont_path)
    
    print("\nConversion Results:")
    print("=" * 60)
    for input_file, output_file in results.items():
        status = "✓" if output_file else "✗"
        output_name = Path(output_file).name if output_file else "Failed"
        print(f"{status} {Path(input_file).name} → {output_name}")


if __name__ == "__main__":
    main()
