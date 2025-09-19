# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import os, tempfile

app = FastAPI()
# allow your Next.js dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup.
# "base.en" is fast English-only; use "small"/"medium"/"large-v3" for multilingual/accuracy.
MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")      # "cpu" or "metal" (if supported) or "cuda"
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")   # int8 is fast on CPU

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = Form("en")):
    # Save audio to a temp file (webm/opus, m4a, wav, mp3 all fine)
    suffix = os.path.splitext(file.filename or "")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,      # "en" for base.en; None/"" for auto with multilingual models
            vad_filter=True,
            beam_size=5,
        )
        text = "".join(s.text for s in segments).strip()
        return JSONResponse({"language": info.language, "duration": info.duration, "text": text})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass