import os
import shutil
from pathlib import Path

# Source directories
SOURCE_DIR = Path(r"C:\Users\Dhanuja\Downloads\dataset\source_chunks")
TARGET_DIR = Path(r"C:\Users\Dhanuja\Downloads\dataset\punk_data")

# Output directories
OUTPUT_DIR = Path(r"C:\Users\Dhanuja\Downloads\dataset\dataset")
OUTPUT_SOURCE = OUTPUT_DIR / "source"
OUTPUT_TARGET = OUTPUT_DIR / "target"

OUTPUT_SOURCE.mkdir(parents=True, exist_ok=True)
OUTPUT_TARGET.mkdir(parents=True, exist_ok=True)

# Get filenames - strip '_synth' from source stems to get base name
source_files = {}
for f in SOURCE_DIR.glob("*.mp3"):
    base_stem = f.stem.replace("_synth", "")  # remove _synth suffix
    source_files[base_stem] = f

# Target stems are already the base name
target_files = {}
for f in TARGET_DIR.glob("*.mp3"):
    target_files[f.stem] = f

# Find common stems
common_stems = set(source_files.keys()) & set(target_files.keys())

print(f"Source files:  {len(source_files)}")
print(f"Target files:  {len(target_files)}")
print(f"Common pairs:  {len(common_stems)}")
print(f"Missing in source: {len(target_files) - len(common_stems)}")
print(f"Missing in target: {len(source_files) - len(common_stems)}")

# Copy matched pairs to output dirs
for i, stem in enumerate(sorted(common_stems), 1):
    src_file = source_files[stem]
    tgt_file = target_files[stem]

    shutil.copy2(src_file, OUTPUT_SOURCE / src_file.name)
    shutil.copy2(tgt_file, OUTPUT_TARGET / tgt_file.name)

    if i % 1000 == 0:
        print(f"  Copied {i}/{len(common_stems)} pairs...")

print(f"\nDone!")
print(f"Output source: {OUTPUT_SOURCE} → {len(list(OUTPUT_SOURCE.iterdir()))} files")
print(f"Output target: {OUTPUT_TARGET} → {len(list(OUTPUT_TARGET.iterdir()))} files")
