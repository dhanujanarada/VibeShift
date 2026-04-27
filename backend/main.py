import os
import sys
import uuid, shutil
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root and backend/ are on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))   # project root → models/, utills/
sys.path.insert(0, str(Path(__file__).parent))           # backend/ → inference/

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from inference.vibeshift_inference import VibeshiftInference

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "Nyarada/vibeshift-checkpoints")
UPLOAD_DIR  = Path("tmp/uploads")
OUTPUT_DIR  = Path("tmp/outputs")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

pipeline: VibeshiftInference | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    # Wipe and recreate tmp dirs on every startup
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print(f"[startup] Loading pipeline from HF repo: {HF_REPO_ID}")
    pipeline = VibeshiftInference(hf_repo_id=HF_REPO_ID)
    print("[startup] Pipeline ready.")

    yield

    # Clean up on shutdown
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        if d.exists():
            shutil.rmtree(d)

app = FastAPI(title="VibeShift API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/transform")
async def transform(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(503, "Model not loaded yet.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Only audio files are supported ({', '.join(sorted(AUDIO_EXTENSIONS))}).")

    # Save uploaded file
    uid = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{uid}_{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run inference
    job_output_dir = OUTPUT_DIR / uid
    try:
        result = pipeline.generate(str(upload_path), str(job_output_dir))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    upload_path.unlink(missing_ok=True)

    return {
        "uid": uid,
        "input_url":  f"/audio/{uid}/input",
        "synth_url":  f"/audio/{uid}/synth",
        "output_url": f"/audio/{uid}/output",
        "sample_rate": result["sample_rate"],
    }

@app.get("/audio/{uid}/{kind}")
def serve_audio(uid: str, kind: str):
    if kind not in ("input", "synth", "output"):
        raise HTTPException(404, "Unknown audio kind.")

    matches = list((OUTPUT_DIR / uid).glob(f"*_{kind}.wav"))
    if not matches:
        raise HTTPException(404, "Audio not found.")

    return FileResponse(
        str(matches[0]),
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename={kind}.wav"}
    )

