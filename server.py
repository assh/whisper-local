# server.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select
from passlib.context import CryptContext
import jwt, os, tempfile, datetime as dt
from typing import Optional

# === Whisper ===
from faster_whisper import WhisperModel

# -----------------------
# App & CORS
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Config
# -----------------------
DB_URL = os.getenv("AUTH_DB_URL", "sqlite:///./auth.db")
JWT_SECRET = os.getenv("AUTH_SECRET", "change-me")
JWT_ALG = "HS256"
JWT_MINS = int(os.getenv("AUTH_TTL_MIN", "60"))
INVITE_CODE = os.getenv("SIGNUP_INVITE_CODE")  # optional

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})

# -----------------------
# Models
# -----------------------
class User(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  email: EmailStr = Field(index=True, unique=True)
  password_hash: str
  name: str = "User"

  # Presets (you can fill later when you wire Trello)
  summary_card: Optional[str] = None
  checklist_card: Optional[str] = None

class UserCreate(BaseModel):
  email: EmailStr
  password: str
  name: Optional[str] = "User"
  inviteCode: Optional[str] = None

class UserLogin(BaseModel):
  email: EmailStr
  password: str

class UserOut(BaseModel):
  id: int
  email: EmailStr
  name: str
  defaults: dict

class TokenOut(BaseModel):
  access_token: str
  token_type: str = "bearer"
  user: UserOut

# -----------------------
# DB init
# -----------------------
def init_db():
  SQLModel.metadata.create_all(engine)

@app.on_event("startup")
def on_startup():
  init_db()

# -----------------------
# Auth helpers
# -----------------------
def hash_pw(p: str) -> str:
  return pwd.hash(p)

def verify_pw(p: str, h: str) -> bool:
  return pwd.verify(p, h)

def make_jwt(user_id: int) -> str:
  now = dt.datetime.utcnow()
  exp = now + dt.timedelta(minutes=JWT_MINS)
  payload = {"sub": str(user_id), "iat": now, "exp": exp}
  return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_jwt(token: str) -> int:
  try:
    data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    return int(data["sub"])
  except jwt.ExpiredSignatureError:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
  except Exception:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def require_user(authorization: Optional[str] = None) -> User:
  if not authorization or not authorization.lower().startswith("bearer "):
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
  token = authorization.split(" ", 1)[1].strip()
  user_id = decode_jwt(token)
  with Session(engine) as s:
    user = s.get(User, user_id)
    if not user:
      raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# -----------------------
# Auth endpoints
# -----------------------
@app.post("/auth/register", response_model=dict)
def register(payload: UserCreate):
  # optional invite guard
  if INVITE_CODE and payload.inviteCode != INVITE_CODE:
    raise HTTPException(status_code=403, detail="Invite code required")

  with Session(engine) as s:
    exists = s.exec(select(User).where(User.email == payload.email)).first()
    if exists:
      raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
      email=payload.email,
      password_hash=hash_pw(payload.password),
      name=payload.name or "User",
    )
    s.add(user)
    s.commit()
    s.refresh(user)

  return {"ok": True, "id": user.id}

@app.post("/auth/login", response_model=TokenOut)
def login(payload: UserLogin):
  with Session(engine) as s:
    user = s.exec(select(User).where(User.email == payload.email)).first()
    if not user or not verify_pw(payload.password, user.password_hash):
      raise HTTPException(status_code=401, detail="Invalid credentials")

  token = make_jwt(user.id)
  return TokenOut(
    access_token=token,
    user=UserOut(
      id=user.id,
      email=user.email,
      name=user.name,
      defaults={
        "summaryCard": user.summary_card,
        "checklistCard": user.checklist_card,
      },
    ),
  )

@app.get("/me", response_model=UserOut)
def me(current: User = Depends(require_user)):
  return UserOut(
    id=current.id,
    email=current.email,
    name=current.name,
    defaults={
      "summaryCard": current.summary_card,
      "checklistCard": current.checklist_card,
    },
  )

# -----------------------
# (Optionally) presets update endpoints
# -----------------------
class PresetsIn(BaseModel):
  summaryCard: Optional[str] = None
  checklistCard: Optional[str] = None

@app.post("/me/presets", response_model=UserOut)
def update_presets(presets: PresetsIn, current: User = Depends(require_user)):
  with Session(engine) as s:
    user = s.get(User, current.id)
    if presets.summaryCard is not None:
      user.summary_card = presets.summaryCard.strip() or None
    if presets.checklistCard is not None:
      user.checklist_card = presets.checklistCard.strip() or None
    s.add(user)
    s.commit()
    s.refresh(user)
    return UserOut(
      id=user.id,
      email=user.email,
      name=user.name,
      defaults={"summaryCard": user.summary_card, "checklistCard": user.checklist_card},
    )

# -----------------------
# Whisper transcription (unchanged)
# -----------------------
MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")      # "cpu" or "metal" or "cuda"
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")   # int8 is fast on CPU
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = Form("en")):
  suffix = os.path.splitext(file.filename or "")[-1] or ".webm"
  with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(await file.read())
    tmp_path = tmp.name

  try:
    segments, info = model.transcribe(
      tmp_path,
      language=language,
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