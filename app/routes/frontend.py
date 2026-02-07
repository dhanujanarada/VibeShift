from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid
import sys
import os

# Add parent directory to path to import inference module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import VibeShiftInference

router = APIRouter()

# Get the app directory
APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent

templates = Jinja2Templates(
    directory=str(APP_DIR / "templates")
)

OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Path to the best model checkpoint
CHECKPOINT_PATH = ROOT_DIR / "notebooks" / "checkpoints" / "paired_training" / "best_model.pt"

# Initialize inference model (lazy loading)
_inference_model = None


def get_inference_model():
    """Lazy load the inference model"""
    global _inference_model
    if _inference_model is None:
        print("Initializing inference model...")
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        _inference_model = VibeShiftInference(str(CHECKPOINT_PATH))
        print("✓ Model loaded successfully!")
    return _inference_model


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@router.post("/transform", response_class=HTMLResponse)
async def transform_audio(
    request: Request,
    audio: UploadFile = File(...)
):
    try:
        # Create unique filenames
        input_id = uuid.uuid4().hex
        input_name = f"{input_id}_input{Path(audio.filename).suffix}"
        output_name = f"{input_id}_rock.wav"
        
        input_path = OUTPUT_DIR / input_name
        output_path = OUTPUT_DIR / output_name
        
        # Save uploaded file
        print(f"Saving uploaded file: {input_name}")
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Get inference model
        model = get_inference_model()
        
        # Transform audio
        print(f"Transforming audio: {input_name}")
        model.transform_audio(
            audio_path=str(input_path),
            output_path=str(output_path),
            target_genre=1,  # Rock
            num_steps=50,
            method="heun"
        )
        
        # Clean up input file
        if input_path.exists():
            input_path.unlink()
        
        print(f"✓ Transformation complete: {output_name}")
        
        return templates.TemplateResponse(
            "partials/download.html",
            {
                "request": request,
                "file_url": f"/outputs/{output_name}",
                "filename": output_name
            }
        )
    
    except Exception as e:
        print(f"Error during transformation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")
