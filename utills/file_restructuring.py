import os
import shutil

src_dir = r"C:\Users\Dhanuja\Downloads\dataset\htdemucs"
dst_dir = r"C:\Users\Dhanuja\Downloads\dataset\instrumentals"  # change this to your preferred output dir

os.makedirs(dst_dir, exist_ok=True)

copied = 0
skipped = 0

for folder_name in os.listdir(src_dir):
    folder_path = os.path.join(src_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    no_vocal_src = os.path.join(folder_path, "no_vocals.mp3")
    
    if os.path.exists(no_vocal_src):
        new_name = f"{folder_name}_instrumental.mp3"
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy2(no_vocal_src, dst_path)
        copied += 1
        print(f"Copied: {new_name}")
    else:
        print(f"Skipped (no file found): {folder_name}")
        skipped += 1

print(f"\nDone! Copied: {copied}, Skipped: {skipped}")