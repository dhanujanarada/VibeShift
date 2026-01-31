# VibeShift

AI-powered music genre transformation using Flow Matching and Diffusion Transformers (DiT).

## Project Overview

VibeShift transforms music from one genre to another using state-of-the-art generative models. It uses:
- **Flow Matching** for smooth interpolation between genre representations
- **DiT (Diffusion Transformer)** for high-quality mel-spectrogram generation
- **FiLM conditioning** for genre control
- **Mel-spectrogram processing** for audio representation

## Setup

### Prerequisites
- Python 3.10.x (3.11 not supported due to dependencies)
- [uv](https://github.com/astral-sh/uv) package manager 
- FFmpeg (for audio processing)

### Installation

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

3. **Set up the environment**
```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\activate


```

## Running the Application

### Frontend (Web Interface)

1. **Start the FastAPI server**
```bash
# Make sure virtual environment is activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the web interface**
Open your browser and navigate to: `http://localhost:8000`

### Training

1. **Prepare the dataset**
Audio files are in the appropriate directories:
- `data/rock_audio_files/` for rock genre
- `data/nonrock_audio_files/` for other genres

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

### Virtual Environment Issues
If you encounter issues with the virtual environment:
```bash
# Remove corrupted venv
rm -rf .venv

# Recreate with uv
uv sync
```

### PyTorch Import Errors
Ensure you're using Python 3.10.x:
```bash
python --version  
```

### FFmpeg Not Found
Install FFmpeg:
- **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`
