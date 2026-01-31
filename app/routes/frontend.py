from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid, shutil

router = APIRouter()

# Get the app directory
APP_DIR = Path(__file__).parent.parent

templates = Jinja2Templates(
    directory=str(APP_DIR / "templates")
)

OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    output_name = f"{uuid.uuid4().hex}_rock.wav"
    output_path = OUTPUT_DIR / output_name

    # TEMP: fake "model" output
    with open(output_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    return templates.TemplateResponse(
        "partials/download.html",
        {
            "request": request,
            "file_url": f"/outputs/{output_name}"
        }
    )
