from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid
import sys
import os
import asyncio
import threading
from functools import partial
from starlette.concurrency import run_in_threadpool
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference2 import VibeShiftInference

router = APIRouter()


APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Directories
INPUT_DIR = APP_DIR / "inputs"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Nyarada/vibeshift-checkpoints",
    filename="best.pt",
    local_dir="test_download",
    token=os.environ.get("HF_TOKEN")
)
print(f"Downloaded to: {path}")

# Model paths
CHECKPOINT_PATH = path
# Use system soundfont (installed via apt fluid-soundfont-gm), fallback to local
_system_sf2 = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
_local_sf2 = ROOT_DIR / "data" / "TimGM6mb.sf2"
SOUNDFONT_PATH = _system_sf2 if _system_sf2.exists() else _local_sf2

# Settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
TRANSFORM_TIMEOUT = 900  # 15 minutes



_model = None
_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print("Loading VibeShift model...")
                _model = VibeShiftInference(
                    checkpoint_path=str(CHECKPOINT_PATH),
                    soundfont_path=str(SOUNDFONT_PATH),
                    device="cpu"
                )
                print("✅ Model ready!\n")
    return _model


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/transform", response_class=HTMLResponse)
async def transform_audio(request: Request, audio: UploadFile = File(...)):
    input_path = temp_dir = None

    try:
        # Validate file
        ext = Path(audio.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")

        # Save uploaded file
        file_id = uuid.uuid4().hex
        input_path = INPUT_DIR / f"{file_id}_input{ext}"

        size = 0
        with open(input_path, "wb") as f:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large (max 50MB)")
                f.write(chunk)

        print(f"\n💾 Uploaded: {input_path.name} ({size/(1024*1024):.1f}MB)")

        # Process audio
        temp_dir = OUTPUT_DIR / file_id
        temp_dir.mkdir(exist_ok=True)

        print(f"🎵 Processing → Classical...")

        result = await asyncio.wait_for(
            run_in_threadpool(
                partial(
                    get_model().process,
                    input_audio_path=str(input_path),
                    target_genre=0,  # Classical
                    output_dir=str(temp_dir),
                    num_steps=50,
                    save_intermediates=False
                )
            ),
            timeout=TRANSFORM_TIMEOUT
        )

        # Move output to main folder
        temp_output = Path(result['output_audio'])
        final_output = OUTPUT_DIR / f"{file_id}_classical.wav"
        temp_output.rename(final_output)

        print(f"✅ Complete: {final_output.name}\n")

        return templates.TemplateResponse(
            "partials/download.html",
            {
                "request": request,
                "file_url": f"/outputs/{final_output.name}",
                "filename": final_output.name
            }
        )

    except asyncio.TimeoutError:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error_message": "Processing timed out. Try a shorter audio clip (under 30 seconds)."},
            status_code=200
        )

    except HTTPException as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error_message": e.detail},
            status_code=200
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error_message": str(e)},
            status_code=200
        )

    finally:
        # Cleanup input file
        try:
            if input_path and input_path.exists():
                input_path.unlink()
        except Exception as cleanup_exc:
            print(f"⚠️ Input cleanup failed: {cleanup_exc}")
        # Cleanup temp output directory
        try:
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"Temp dir cleanup failed: {cleanup_exc}")


@router.get("/cleanup")
async def cleanup(background_tasks: BackgroundTasks):
    """Delete output files older than 24 hours"""
    def clean():
        now = time.time()
        count = 0
        for f in OUTPUT_DIR.glob("*.wav"):
            if now - f.stat().st_mtime > 86400:  # 24 hours
                f.unlink()
                count += 1
        print(f"🗑️ Cleaned {count} old files")

    background_tasks.add_task(clean)
    return {"status": "Cleanup scheduled"}