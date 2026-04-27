
# VibeShift

VibeShift is an AI-powered tool for music genre transformation using Flow Matching and Diffusion Transformers (DiT). It provides both a backend API (FastAPI) and a Svelte-based frontend for interactive use.

## Features

- Audio-to-audio genre transfer using state-of-the-art generative models
- Batch and single-file processing
- FastAPI backend for inference and file handling
- Svelte frontend for user interaction and visualization
- Training scripts and configuration for model development

## Requirements

- Windows
- Python 3.10 (required, managed by `uv`)
- Node.js (for frontend)
- Git
- FFmpeg (for audio processing)
- Internet connection (for downloading model checkpoints and dependencies)

## Declaration
### AI Usage Declaration

AI language models were used during the development of this project to assist 
with coding tasks, including generating boilerplate code, suggesting 
implementations for standard components, debugging errors, and explaining 
unfamiliar libraries or APIs. All AI-generated code was reviewed, tested, 
and validated by the author before integration. Core architectural decisions, 
model design, and research implementation logic were authored by the author 
independently.

### Citations
Demcus was used to pre process the vocal stems out of the audio tracks.
```bash
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```
## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd VibeShift
```

### 2. Install Python environment and dependencies

Install [uv](https://github.com/astral-sh/uv) if not already installed:

Create and activate the Python 3.10 environment, and install dependencies:

```bash
uv python install 3.10
uv sync
.venv\Scripts\activate  # Windows

```

### 3. Install FFmpeg

- **Windows:** `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)

### 4. Install FluidSynth

FluidSynth is required for MIDI-to-audio synthesis and must be installed at the system level.

- **Windows:** `choco install fluidsynth` or download from [fluidsynth.org](https://www.fluidsynth.org)
- **Linux:** `sudo apt install fluidsynth`
- **macOS:** `brew install fluidsynth`

Ensure `fluidsynth` is on your `PATH` after installation.

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the backend API

From the project root:

```bash
uv run uvicorn backend.main:app --reload
```

> **Important:** Always use `uv run` (not a bare `uvicorn` call) so the correct `.venv` Python and all installed packages are used. Running `uvicorn` directly may resolve to system Python and fail with `ModuleNotFoundError`.
The API will running on  http://localhost:8000

### 2. Start the frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will running on [http://localhost:5173](http://localhost:5173/)

## Usage

- Upload an audio file (WAV, MP3, FLAC, OGG, M4A) via the frontend.
- The backend will process the file and return genre-transformed audio.
- Download or preview the results in the browser.

## Configuration


- Model checkpoints are downloaded automatically from HuggingFace (see `backend/main.py`).

- No `.env` file is required by default. The backend uses the `HF_REPO_ID` environment variable to specify the HuggingFace repo for checkpoints (default: `Nyarada/vibeshift-checkpoints`).

## Troubleshooting

- If dependencies fail to install, delete `.venv` and run `uv sync` again.
- If you see `ModuleNotFoundError: No module named 'dac'` or similar, make sure you are running the server with `uv run uvicorn backend.main:app --reload` from the **project root**, not with a bare `uvicorn` command.
- If you see `AttributeError: module 'dac' has no attribute 'model'`, ensure `dac>=0.4.3` is **not** listed in `pyproject.toml`. Only `descript-audio-codec` should provide the `dac` module.
- If FluidSynth is not found, make sure it is installed and `fluidsynth` is on your system `PATH`.
- If you see PyTorch or FFmpeg errors, ensure the correct Python version and that FFmpeg is installed and on your PATH.
- If port 8000 is in use, run the backend on a different port:

```bash
uv run uvicorn backend.main:app --reload --port 8080
```

## Project Structure

- `backend/` — FastAPI backend and inference code
- `frontend/` — Svelte frontend
- `configs/` — Model and training configuration files
- `data/` — Datasets and audio files
- `models/`, `utills/`, `Scripts/`,  `Training/` — Model code, utilities, and training scripts
- `test/` — Test notebooks and scripts

If you need more details or have questions, please refer to the code comments 
uv python install 3.10
