# VibeShift

AI-powered music genre transformation using Flow Matching and Diffusion Transformers (DiT).


## Project Overview

VibeShift transforms music from one genre to another using state-of-the-art generative models. It uses:
- **Flow Matching** for smooth interpolation between genre representations
- **DiT (Diffusion Transformer)** for high-quality mel-spectrogram generation
- **FiLM conditioning** for genre control
- **Mel-spectrogram processing** for audio representation

### Prerequisites
- Git
- FFmpeg (for audio processing)
- Internet connection (for downloading dependencies)

### Quick Start 

1. **Clone the repository**
```bash
git clone <repository-url>
cd VibeShift
```

2. **Install uv (if not already installed)**
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"


```

3. **Install Python 3.10 and setup environment**
```bash
# Install Python 3.10 (uv will manage this)
uv python install 3.10

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\activate


```

### Verify Installation
```bash
# Check Python version (should be 3.10.x)
python --version

# Test PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Running the Application

```bash
# Windows
.venv\Scripts\activate
uvicorn app.main:app --reload

```

Visit: `http://localhost:8000`


### Training

1. **Prepare the dataset**
Audio files are in the appropriate directories:
- `data/rock_audio_files/` for rock genre
- `data/classical/` for classical

2. **Run training**
```bash
python train.py
```

Configuration are in:
- `configs/dit.yaml` - DiT model architecture
- `configs/flow.yaml` - Flow matching parameters
- `configs/genres.yaml` - Genre mappings

### Inference

**Run inference on a single audio file:**
```bash

```

## Project Structure

```
VibeShift/
├── app/                    # FastAPI web application
│   ├── main.py            # Application entry point
│   ├── routes/            # API routes
│   ├── static/            # CSS, JS files
│   └── templates/         # HTML templates
├── models/                # Model architectures
│   ├── dit.py            # Diffusion Transformer
│   ├── flow.py           # Flow Matching implementation
│   └── film_conditioner.py
├── training/              # Training scripts
│   └── training.py       # Main training loop
├── utills/                # Utility functions
│   ├── mel.py            # Mel-spectrogram processing
│   └── embedding.py      # Genre embeddings
├── data/                  # Dataset (not tracked in git)
├── configs/               # Configuration files
├── checkpoints/           # Model checkpoints (not tracked)
└── notebooks/            # Jupyter notebooks for experiments
```

## Configuration

### Model Configuration (`configs/dit.yaml`)
```yaml
dit_model:
  patch_height: 10      # Mel-spectrogram patch height
  patch_width: 64       # Mel-spectrogram patch width
  embed_dim: 512        # Embedding dimension
  num_blocks: 12        # Number of transformer blocks
  num_heads: 8          # Attention heads
  num_genres: 10        # Number of supported genres
```


## Development

### Testing
```bash
# Test DiT model
jupyter notebook test/test_dit_model.ipynb

# Test Flow Matching
jupyter notebook test/test_flow_matching.ipynb
```


## Troubleshooting

### "uv sync fails" or "Dependencies won't install"
1. **Delete the virtual environment and retry:**
   ```bash
   # Windows
   Remove-Item -Recurse -Force .venv
   
   # macOS/Linux
   rm -rf .venv
   ```

2. **Run the setup script again:**
   ```bash
   # Windows
   setup.bat
   
   # macOS/Linux
   ./setup.sh
   ```

3. **Or manually:**
   ```bash
   uv python install 3.10
   uv sync
   ```

### Virtual Environment Issues
If the virtual environment becomes corrupted:
```bash
# Remove corrupted venv
rm -rf .venv  # or Remove-Item -Recurse -Force .venv on Windows

# Recreate with uv
uv python install 3.10
uv sync
```

### PyTorch Import Errors
**Error: `RuntimeError: function 'istft' already has a docstring`**

This usually means the environment is corrupted. Solution:
```bash
# Delete .venv folder completely
# Re-run setup script or uv sync
```

Ensure you're using Python 3.10.x:
```bash
python --version  # Should show 3.10.x
```

### FFmpeg Not Found
Install FFmpeg:
- **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### Port Already in Use
If port 8000 is already in use:
```bash
# Use a different port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

### Module Not Found Errors
Make sure the virtual environment is activated:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```
