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
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
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

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Set up the environment**
```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
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

1. **Prepare your dataset**
Place audio files in the appropriate directories:
- `data/rock_audio_files/` for rock genre
- `data/nonrock_audio_files/` for other genres

2. **Run training**
```bash
python train.py
```

Training configuration can be modified in:
- `configs/dit.yaml` - DiT model architecture
- `configs/flow.yaml` - Flow matching parameters
- `configs/genres.yaml` - Genre mappings

### Inference

**Run inference on a single audio file:**
```bash
python inference.py --input path/to/audio.mp3 --target-genre rock
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

### Genre Mappings (`configs/genres.yaml`)
Define source and target genre mappings for transformation.

## Development

### Testing Models
```bash
# Test DiT model
jupyter notebook test/test_dit_model.ipynb

# Test Flow Matching
jupyter notebook test/test_flow_matching.ipynb
```

### Adding New Features
1. Create feature branch: `git checkout -b feature-name`
2. Make changes and test thoroughly
3. Commit: `git commit -m "Description"`
4. Push: `git push origin feature-name`

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
python --version  # Should show 3.10.x
```

### FFmpeg Not Found
Install FFmpeg:
- **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## License

[Your License Here]

## Contributors

[Your Contributors Here]