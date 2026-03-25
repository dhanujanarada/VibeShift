
# VibeShift

VibeShift is an AI-powered tool for music genre transformation using Flow Matching and Diffusion Transformers (DiT). It provides both a backend API (FastAPI) and a Svelte-based frontend for interactive use.

## Features

- Audio-to-audio genre transfer using state-of-the-art generative models
- Batch and single-file processing
- FastAPI backend for inference and file handling
- Svelte frontend for user interaction and visualization
- Training scripts and configuration for model development

## Requirements

- Windows, macOS, or Linux
- Python 3.10 (required, managed by `uv`)
- Node.js (for frontend)
- Git
- FFmpeg (for audio processing)
- Internet connection (for downloading model checkpoints and dependencies)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd VibeShift
```

### 2. Install Python environment and dependencies

Install [uv](https://github.com/astral-sh/uv) if not already installed:

```bash
# Windows (PowerShell)

```

Create and activate the Python 3.10 environment, and install dependencies:

```bash
uv python install 3.10
uv sync
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

### 3. Install FFmpeg

- **Windows:** `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the backend API

From the project root:

```bash
.venv\Scripts\activate  # or source .venv/bin/activate
uvicorn backend.main:app --reload
```

The API will be available at http://localhost:8000

### 2. Start the frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:5173

## Usage

- Upload an audio file (WAV, MP3, FLAC, OGG, M4A) via the frontend.
- The backend will process the file and return genre-transformed audio.
- Download or preview the results in the browser.

## Configuration

- Model and training configuration files are in the `configs/` directory:
  - `dit.yaml`: DiT model architecture
  - `flow.yaml`: Flow matching parameters
  - `genres.yaml`: Genre mappings

- Model checkpoints are downloaded automatically from HuggingFace (see `backend/main.py`).

- No `.env` file is required by default. The backend uses the `HF_REPO_ID` environment variable to specify the HuggingFace repo for checkpoints (default: `Nyarada/vibeshift-checkpoints`).

## Training

1. Prepare your dataset in the `data/` directory.
2. Edit configuration files in `configs/` as needed.
3. Run training:

```bash
python training/training.py
```

## Testing

- Test notebooks are in the `test/` directory.
- Example:
  - `jupyter notebook test/test_dit_model.ipynb`
  - `jupyter notebook test/test_flow_matching.ipynb`

## Troubleshooting

- If dependencies fail to install, delete `.venv` and run `uv sync` again.
- If you see PyTorch or FFmpeg errors, ensure the correct Python version and that FFmpeg is installed and on your PATH.
- If port 8000 is in use, run the backend on a different port:

```bash
uvicorn backend.main:app --reload --port 8080
```

## Project Structure

- `backend/` — FastAPI backend and inference code
- `frontend/` — Svelte frontend
- `configs/` — Model and training configuration files
- `data/` — Datasets and audio files
- `models/`, `utills/`, `training/` — Model code, utilities, and training scripts
- `test/` — Test notebooks and scripts

## License

See `LICENSE` file for details.

If you need more details or have questions, please refer to the code comments or open an issue.
uv python install 3.10
