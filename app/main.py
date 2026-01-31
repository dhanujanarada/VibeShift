from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.routes.frontend import router

app = FastAPI()

# Get the app directory
APP_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(APP_DIR / "outputs")), name="outputs")

app.include_router(router)
