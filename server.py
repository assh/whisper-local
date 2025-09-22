# server.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select
from passlib.context import CryptContext
import jwt, os, tempfile, datetime as dt
from typing import Optional, Tuple
import httpx
from dotenv import load_dotenv
from pathlib import Path

# === Whisper ===
from faster_whisper import WhisperModel

# -----------------------
# App & CORS
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # add your deployed origin here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load environment files (.env, .env.local) next to server.py
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local", override=True)

# -----------------------
# Config
# -----------------------
DB_URL = os.getenv("AUTH_DB_URL", "sqlite:///./auth.db")
JWT_SECRET = os.getenv("AUTH_SECRET", "change-me")
JWT_ALG = "HS256"
JWT_MINS = int(os.getenv("AUTH_TTL_MIN", "60"))
INVITE_CODE = os.getenv("SIGNUP_INVITE_CODE")  # optional invite guard

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})

# -----------------------

# -----------------------
# Models
# -----------------------
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: EmailStr = Field(index=True, unique=True)
    password_hash: str
    name: str = "User"

    # Presets (defaults for UI)
    summary_card: Optional[str] = None
    checklist_card: Optional[str] = None

    # Per-user Trello credentials (MVP: stored as plain text)
    trello_api_key: Optional[str] = None
    trello_token: Optional[str] = None

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

def require_user(authorization: str | None = Header(None)) -> User:
    """
    Dependency to extract and validate the current user from a Bearer token.
    """
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
            defaults={"summaryCard": user.summary_card, "checklistCard": user.checklist_card},
        ),
    )

@app.get("/me", response_model=UserOut)
def me(current: User = Depends(require_user)):
    return UserOut(
        id=current.id,
        email=current.email,
        name=current.name,
        defaults={"summaryCard": current.summary_card, "checklistCard": current.checklist_card},
    )

# -----------------------
# Presets (defaults) update endpoints
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
# Per-user Trello creds (encrypted)
# -----------------------
class TrelloCredsIn(BaseModel):
    apiKey: str
    token: str

@app.post("/me/trello", response_model=dict)
def set_trello_creds(creds: TrelloCredsIn, current: User = Depends(require_user)):
    with Session(engine) as s:
        u = s.get(User, current.id)
        u.trello_api_key = creds.apiKey.strip()
        u.trello_token = creds.token.strip()
        s.add(u); s.commit()
    return {"ok": True}

@app.get("/me/trello", response_model=dict)
def get_trello_creds_masked(current: User = Depends(require_user)):
    key = (current.trello_api_key or "").strip()
    tok = (current.trello_token or "").strip()
    return {
        "apiKey": (key[:4] + "…" + key[-4:]) if key else None,
        "token": (tok[:4] + "…" + tok[-4:]) if tok else None,
    }

@app.delete("/me/trello", response_model=dict)
def clear_trello_creds(current: User = Depends(require_user)):
    with Session(engine) as s:
        u = s.get(User, current.id)
        u.trello_api_key = None
        u.trello_token = None
        s.add(u); s.commit()
    return {"ok": True}

def get_trello_creds(current: User) -> Tuple[str, str]:
    # Per-user first
    key = (current.trello_api_key or "").strip()
    tok = (current.trello_token or "").strip()
    # Optional global fallback (useful for admin/testing)
    if not key:
        key = (os.getenv("TRELLO_KEY") or "").strip()
    if not tok:
        tok = (os.getenv("TRELLO_TOKEN") or "").strip()
    if not key or not tok:
        raise HTTPException(status_code=500, detail="Missing Trello key/token for this user (or env fallback).")
    return key, tok

# -----------------------
# Trello helpers & routes
# -----------------------
def parse_card_id(input_val: str) -> Optional[str]:
    if not input_val:
        return None
    t = input_val.strip()
    import re
    m = re.search(r"trello\.com\/c\/([A-Za-z0-9]+)", t, re.I)
    if m:
        return m.group(1)  # shortLink from URL
    if len(t) == 24 and all(c in "0123456789abcdefABCDEF" for c in t):
        return t  # full 24-char id
    if 6 <= len(t) <= 12:
        return t  # shortLink typed
    return None

@app.get("/trello/ping", response_model=dict)
def trello_ping(current: User = Depends(require_user)):
    key, tok = get_trello_creds(current)
    url = f"https://api.trello.com/1/members/me?key={key}&token={tok}"
    with httpx.Client(timeout=10) as client:
        r = client.get(url)
        data = r.json()
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=data.get("message") or "Trello auth failed")
    return {"ok": True, "me": {k: data.get(k) for k in ("id", "username", "fullName")}}

class CommentIn(BaseModel):
    cardInput: str
    text: str

@app.post("/trello/comment", response_model=dict)
def trello_comment(payload: CommentIn, current: User = Depends(require_user)):
    key, tok = get_trello_creds(current)
    card_id = parse_card_id(payload.cardInput)
    if not card_id:
        raise HTTPException(status_code=400, detail="Provide a valid Trello card URL, short link, or ID.")
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Summary text is required.")
    url = f"https://api.trello.com/1/cards/{card_id}/actions/comments"
    params = {"key": key, "token": tok, "text": payload.text.strip()}
    with httpx.Client(timeout=15) as client:
        r = client.post(url, params=params)
        data = r.json()
    if r.status_code != 200:
        msg = data.get("message") or data.get("error") or f"Trello error {r.status_code}"
        raise HTTPException(status_code=r.status_code, detail=msg)
    return {"ok": True, "data": data}

class ChecklistMergeIn(BaseModel):
    cardInput: str
    items: list[str]
    checklistName: Optional[str] = "Next steps"

@app.post("/trello/checklist/merge", response_model=dict)
def trello_checklist_merge(payload: ChecklistMergeIn, current: User = Depends(require_user)):
    key, tok = get_trello_creds(current)
    card_id = parse_card_id(payload.cardInput)
    if not card_id:
        raise HTTPException(status_code=400, detail="Provide a valid Trello card URL, short link, or ID.")
    items = [s.strip() for s in (payload.items or []) if s and s.strip()]
    if not items:
        raise HTTPException(status_code=400, detail="No checklist items provided.")

    # 1) Get existing checklists on the card
    url_lists = f"https://api.trello.com/1/cards/{card_id}/checklists"
    params = {"key": key, "token": tok}
    with httpx.Client(timeout=15) as client:
        rl = client.get(url_lists, params=params)
        list_data = rl.json()
        if rl.status_code != 200:
            raise HTTPException(status_code=rl.status_code, detail=list_data.get("message") or "Failed to read card checklists.")
        # Assume at most 1 checklist; create if none
        checklist = list_data[0] if isinstance(list_data, list) and list_data else None
        if not checklist:
            url_create = "https://api.trello.com/1/checklists"
            rc = client.post(url_create, params={"key": key, "token": tok, "idCard": card_id, "name": payload.checklistName or "Checklist"})
            create_data = rc.json()
            if rc.status_code != 200:
                raise HTTPException(status_code=rc.status_code, detail=create_data.get("message") or "Failed to create checklist.")
            checklist = create_data

        checklist_id = checklist["id"]
        existing = checklist.get("checkItems") or []
        existing_names = set((it.get("name") or "").strip().lower() for it in existing)

        # 2) Merge only: add items that don't exist
        added, skipped = 0, 0
        for name in items:
            keyname = name.lower()
            if keyname in existing_names:
                skipped += 1
                continue
            url_add = f"https://api.trello.com/1/checklists/{checklist_id}/checkItems"
            ra = client.post(url_add, params={"key": key, "token": tok, "name": name, "pos": "bottom", "checked": "false"})
            if ra.status_code == 200:
                added += 1
                existing_names.add(keyname)
            else:
                # best-effort; continue others
                pass

    return {"ok": True, "mode": "merge", "checklistId": checklist_id, "added": added, "skipped": skipped}

@app.get("/debug/env")
def debug_env():
    # DO NOT enable in production; this is for local troubleshooting only.
    def masked(v: str | None) -> str | None:
        if not v:
            return None
        return (v[:4] + "…" + v[-4:]) if len(v) > 8 else v
    return {
        "AUTH_DB_URL": os.getenv("AUTH_DB_URL"),
        "AUTH_TTL_MIN": os.getenv("AUTH_TTL_MIN"),
        "TRELLO_KEY_set": bool(os.getenv("TRELLO_KEY")),
        "TRELLO_TOKEN_set": bool(os.getenv("TRELLO_TOKEN"))
    }

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
            language=language,      # "en" for base.en; None/"" for auto (multilingual models)
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