"""
Download model checkpoints from cloud storage.

Supports:
- Hugging Face Hub
- Google Drive
- AWS S3
"""

import os
from pathlib import Path
import urllib.request
import shutil


CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "checkpoints1"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best.pt"

# Define checkpoint URLs (update these with your hosting URLs)
CHECKPOINT_URLS = {
    "huggingface": "https://huggingface.co/YOUR_USERNAME/vibeshift/resolve/main/best.pt",
    "google_drive": "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download",
}


def download_checkpoint(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download checkpoint file from URL.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        chunk_size: Download chunk size in bytes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading checkpoint from {url}...")
    print(f"Saving to {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Checkpoint downloaded successfully ({output_path.stat().st_size / (1024**2):.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error downloading checkpoint: {e}")
        return False


def setup_checkpoint():
    """Download checkpoint if it doesn't exist."""
    if CHECKPOINT_FILE.exists():
        print(f"✓ Checkpoint already exists: {CHECKPOINT_FILE}")
        return True
    
    print(f"Checkpoint not found: {CHECKPOINT_FILE}")
    print("Attempting to download...")
    
    # Try Hugging Face first
    for source, url in CHECKPOINT_URLS.items():
        if url.startswith("https://huggingface"):
            print(f"\nTrying {source}...")
            if download_checkpoint(url, CHECKPOINT_FILE):
                return True
    
    print("\n✗ Failed to download checkpoint")
    print(f"Please download manually from:")
    print(f"  - Hugging Face: {CHECKPOINT_URLS.get('huggingface', 'N/A')}")
    print(f"  - Google Drive: {CHECKPOINT_URLS.get('google_drive', 'N/A')}")
    print(f"And save to: {CHECKPOINT_FILE}")
    return False


if __name__ == "__main__":
    setup_checkpoint()
