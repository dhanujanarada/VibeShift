FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    fluidsynth \
    fluid-soundfont-gm \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./

# Install Python dependencies via pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    jinja2 \
    python-multipart \
    huggingface-hub \
    torch \
    torchaudio \
    descript-audio-codec \
    descript-audiotools \
    librosa \
    soundfile \
    omegaconf \
    pydub \
    tqdm \
    ffmpeg-python \
    basic-pitch \
    pretty_midi \
    mido \
    pyloudnorm \
    numpy \
    scipy \
    resampy \
    vocos \
    transformers

# Copy app code
COPY . .

# Create required directories
RUN mkdir -p app/inputs app/outputs

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
