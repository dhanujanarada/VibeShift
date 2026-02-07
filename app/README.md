# VibeShift Web Application

Transform any music to rock with AI-powered genre transformation!

## Features

- 🎸 **Genre Transformation**: Convert any audio file to rock style using flow matching and diffusion transformers
- 🎵 **Audio Preview**: Listen to transformed audio directly in the browser
- ⬇️ **Download**: Download your transformed audio files
- 🎨 **Beautiful UI**: Glassmorphism design with 3D visualizations

## Setup

### Prerequisites

- Python 3.10 (required for compatibility)
- CUDA-capable GPU (recommended) or CPU

### Installation

1. Install dependencies:
```bash
pip install -e .
```

2. Make sure you have a trained model checkpoint at:
```
notebooks/checkpoints/paired_training/best_model.pt
```

### Running the Application

**Option 1: Using the run script (recommended)**
```bash
python run_app.py
```

**Option 2: Using uvicorn directly**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: `http://127.0.0.1:8000`

## Usage

1. Open your browser and navigate to `http://127.0.0.1:8000`
2. Click "Choose an audio file" and select your audio file (WAV, MP3, or FLAC)
3. Click "🎸 Transform to Rock"
4. Wait for the transformation to complete (this may take a minute)
5. Listen to the preview and download your transformed audio!

## Model Architecture

- **DiT (Diffusion Transformer)**: Transformer-based architecture with RoPE embeddings and FiLM conditioning
- **Flow Matching**: Continuous normalizing flows for smooth genre transformation
- **DAC Codec**: Neural audio codec for high-quality latent space representations

## File Structure

```
app/
├── main.py              # FastAPI application entry point
├── routes/
│   └── frontend.py      # Route handlers for web interface
├── templates/           # HTML templates (Jinja2)
│   ├── base.html
│   ├── index.html
│   └── partials/
│       ├── upload_form.html
│       ├── loading.html
│       └── download.html
├── static/
│   └── css/
│       └── style.css    # Glassmorphism styling
└── outputs/             # Temporary storage for transformed audio

inference.py             # Main inference pipeline
models/                  # Model definitions (DiT, Flow Matching)
configs/                 # Configuration files (YAML)
```

## Technical Details

### Inference Pipeline

1. **Audio Loading**: Load and preprocess audio (resample, convert to mono)
2. **DAC Encoding**: Encode audio to latent space (768-dim embeddings)
3. **Flow Matching**: Transform latent space using trained flow matching model
4. **DAC Decoding**: Decode transformed latents back to audio
5. **Output**: Save and serve the transformed audio

### Model Configuration

- Input: DAC latent embeddings (B, T, 768)
- Embedding dimension: 256
- Transformer blocks: 4
- Attention heads: 8
- Target genres: 0=classical, 1=rock, 2=unknown
- Sampling: Heun's method with 50 steps

## Troubleshooting

### Model loading errors
- Ensure the checkpoint file exists at the correct path
- Check that the checkpoint was saved with the correct state dict keys

### CUDA out of memory
- Reduce the audio length
- Use CPU inference by setting device="cpu" in inference.py
- Process audio in smaller chunks

### Audio quality issues
- Ensure input audio is high quality (44.1kHz or higher recommended)
- Try increasing the number of sampling steps (default: 50)
- Experiment with different sampling methods (euler vs heun)

## Credits

Built with:
- FastAPI
- PyTorch
- Descript Audio Codec (DAC)
- HTMX
- Three.js

---

Made with ❤️ for music lovers and AI enthusiasts!
