# server.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select
from passlib.context import CryptContext
import jwt, os, tempfile, datetime as dt
from typing import Optional, Tuple, List, Literal
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

    # --- Notion (MVP, page-based like Trello cards) ---
    notion_api_key: Optional[str] = None
    notion_summary_page: Optional[str] = None   # full URL or page ID
    notion_checklist_page: Optional[str] = None # full URL or page ID

# ---- replace your existing UserCreate with this ----
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = "User"
    inviteCode: Optional[str] = None

    # NEW: allow setting Notion during registration (MVP)
    notionApiKey: Optional[str] = None
    notionSummaryPage: Optional[str] = None
    notionChecklistPage: Optional[str] = None

# ---- add this helper near your models/helpers (once) ----
def build_defaults(u: "User") -> dict:
    return {
        # Trello (existing)
        "summaryCard": u.summary_card,
        "checklistCard": u.checklist_card,
        # Notion (new)
        "notionSummaryPage": u.notion_summary_page,
        "notionChecklistPage": u.notion_checklist_page,
    }

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
# ---- replace your /auth/register with this version ----
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

            # NEW: capture Notion creds/presets at signup (MVP: plain text)
            notion_api_key=(payload.notionApiKey or None),
            notion_summary_page=(payload.notionSummaryPage or None),
            notion_checklist_page=(payload.notionChecklistPage or None),
        )
        s.add(user)
        s.commit()
        s.refresh(user)
        return {"ok": True, "id": user.id}

# ---- replace your /auth/login with this version ----
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
            defaults=build_defaults(user),
        ),
    )

# ---- replace your /me with this version ----
@app.get("/me", response_model=UserOut)
def me(current: User = Depends(require_user)):
    return UserOut(
        id=current.id,
        email=current.email,
        name=current.name,
        defaults=build_defaults(current),
    )

# -----------------------
# Presets (defaults) update endpoints
# -----------------------
class PresetsIn(BaseModel):
    summaryCard: Optional[str] = None
    checklistCard: Optional[str] = None

# ---- replace your /me/presets return with this block ----
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
            defaults=build_defaults(user),  # now includes Notion fields too
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
# Notion helpers & routes (PAGE-based, mirrors Trello "card" flow)
# -----------------------
import re

# Extract a Notion page ID (with or without hyphens) from a URL or raw id
def parse_notion_page_id(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    t = val.strip()
    # match a 32-hex id with or without hyphens at the end of URL or standalone
    m = re.search(r"([0-9a-fA-F]{32})$", t.replace('-', ''))
    if m:
        raw = m.group(1).lower()
        # re-hyphenate to 8-4-4-4-12
        return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"
    # handle already hyphenated uuid in string
    m2 = re.search(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", t)
    if m2:
        return m2.group(1).lower()
    return None

class NotionCredsIn(BaseModel):
    apiKey: str
    summaryPage: Optional[str] = None   # URL or page id
    checklistPage: Optional[str] = None # URL or page id

class NotionSummaryIn(BaseModel):
    pageInput: Optional[str] = None  # override; otherwise use preset
    title: Optional[str] = None
    summary: str
    blockers: Optional[str] = None
    date: Optional[str] = None  # ISO date (YYYY-MM-DD)
    trelloCard: Optional[str] = None

class NotionChecklistIn(BaseModel):
    pageInput: Optional[str] = None  # override; otherwise use preset
    items: List[str]
    dedupe: bool = True  # attempt to avoid duplicates by existing to_do text

def get_notion_creds_and_pages(current: User) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (api_key, summary_page_id, checklist_page_id) using per-user values with optional env fallbacks.
    For env fallbacks, support NOTION_SUMMARY_PAGE / NOTION_CHECKLIST_PAGE.
    """
    api_key = (current.notion_api_key or os.getenv("NOTION_API_KEY") or "").strip()
    summary_page = parse_notion_page_id(current.notion_summary_page or os.getenv("NOTION_SUMMARY_PAGE"))
    checklist_page = parse_notion_page_id(current.notion_checklist_page or os.getenv("NOTION_CHECKLIST_PAGE"))
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing Notion API key for this user (or env fallback).")
    return api_key, summary_page, checklist_page

def notion_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

@app.post("/me/notion", response_model=dict)
def set_notion_creds(creds: NotionCredsIn, current: User = Depends(require_user)):
    with Session(engine) as s:
        u = s.get(User, current.id)
        u.notion_api_key = creds.apiKey.strip()
        if creds.summaryPage is not None:
            u.notion_summary_page = creds.summaryPage.strip() or None
        if creds.checklistPage is not None:
            u.notion_checklist_page = creds.checklistPage.strip() or None
        s.add(u); s.commit()
    return {"ok": True}

@app.get("/me/notion", response_model=dict)
def get_notion_creds_masked(current: User = Depends(require_user)):
    k = (current.notion_api_key or "").strip()
    return {
        "apiKey": (k[:4] + "…" + k[-4:]) if k else None,
        "summaryPage": current.notion_summary_page,
        "checklistPage": current.notion_checklist_page,
    }

@app.delete("/me/notion", response_model=dict)
def clear_notion_creds(current: User = Depends(require_user)):
    with Session(engine) as s:
        u = s.get(User, current.id)
        u.notion_api_key = None
        u.notion_summary_page = None
        u.notion_checklist_page = None
        s.add(u); s.commit()
    return {"ok": True}

@app.get("/notion/ping", response_model=dict)
def notion_ping(current: User = Depends(require_user)):
    token, summary_page, checklist_page = get_notion_creds_and_pages(current)
    with httpx.Client(timeout=15) as client:
        r = client.get("https://api.notion.com/v1/users/me", headers=notion_headers(token))
        try:
            me = r.json()
        except Exception:
            me = None
        if r.status_code != 200:
            msg = (isinstance(me, dict) and (me.get("message") or me.get("error"))) or "Notion auth failed"
            raise HTTPException(status_code=r.status_code, detail=msg)
        # Optionally verify page access if presets exist
        pages_ok = {}
        for label, pid in (("summary", summary_page), ("checklist", checklist_page)):
            if not pid:
                pages_ok[label] = None
                continue
            rp = client.get(f"https://api.notion.com/v1/pages/{pid}", headers=notion_headers(token))
            pages_ok[label] = (rp.status_code == 200)
    return {"ok": True, "me": {"name": (me or {}).get("name"), "id": (me or {}).get("id")}, "pages": pages_ok}

@app.post("/notion/summary", response_model=dict)
def notion_summary(payload: NotionSummaryIn, current: User = Depends(require_user)):
    token, preset_summary_page, _ = get_notion_creds_and_pages(current)
    page_id = parse_notion_page_id(payload.pageInput) or preset_summary_page
    if not page_id:
        raise HTTPException(status_code=400, detail="No Notion page provided for summary (set preset or pass pageInput).")
    # Compose content blocks similar to Trello comment
    blocks = []
    title = payload.title or f"Daily Summary {dt.datetime.utcnow().date().isoformat()}"
    blocks.append({"type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": title}}]}})
    if payload.summary:
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"type":"text","text":{"content": payload.summary[:1900]}}]}})
    if payload.blockers:
        blocks.append({"type": "heading_3", "heading_3": {"rich_text": [{"type":"text","text":{"content":"Blockers"}}]}})
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"type":"text","text":{"content": payload.blockers[:1900]}}]}})
    if payload.trelloCard:
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"type":"text","text":{"content":"Trello: "}}, {"type":"text","text":{"content": payload.trelloCard, "link": {"url": payload.trelloCard}}}]}})
    with httpx.Client(timeout=20) as client:
        r = client.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=notion_headers(token),
            json={"children": blocks},
        )
        try:
            data = r.json()
        except Exception:
            data = None
    if r.status_code not in (200, 201):
        msg = (isinstance(data, dict) and (data.get("message") or data.get("error"))) or "Failed to append to Notion page."
        raise HTTPException(status_code=r.status_code, detail=msg)
    return {"ok": True, "pageId": page_id, "appended": len(blocks)}

@app.post("/notion/checklist/merge", response_model=dict)
def notion_checklist_merge(payload: NotionChecklistIn, current: User = Depends(require_user)):
    token, _, preset_checklist_page = get_notion_creds_and_pages(current)
    page_id = parse_notion_page_id(payload.pageInput) or preset_checklist_page
    if not page_id:
        raise HTTPException(status_code=400, detail="No Notion page provided for checklist (set preset or pass pageInput).")
    items = [s.strip() for s in (payload.items or []) if s and s.strip()]
    if not items:
        raise HTTPException(status_code=400, detail="No checklist items provided.")
    dedupe_set = set()
    if payload.dedupe:
        # fetch first 200 child blocks and collect existing to_do text to avoid duplicates
        with httpx.Client(timeout=20) as client:
            rr = client.get(f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=200", headers=notion_headers(token))
            if rr.status_code == 200:
                try:
                    children = rr.json().get("results", [])
                except Exception:
                    children = []
                for b in children:
                    if b.get("type") == "to_do":
                        txt = "".join([span.get("plain_text","") for span in b.get("to_do",{}).get("rich_text",[])]).strip().lower()
                        if txt:
                            dedupe_set.add(txt)
    new_blocks = []
    added, skipped = 0, 0
    for it in items:
        key = it.lower()
        if key in dedupe_set:
            skipped += 1
            continue
        new_blocks.append({"type":"to_do","to_do":{"rich_text":[{"type":"text","text":{"content":it}}],"checked":False}})
        added += 1
    if not new_blocks:
        return {"ok": True, "mode": "merge", "pageId": page_id, "added": 0, "skipped": skipped}
    with httpx.Client(timeout=20) as client:
        r = client.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=notion_headers(token),
            json={"children": new_blocks},
        )
        try:
            data = r.json()
        except Exception:
            data = None
    if r.status_code not in (200, 201):
        msg = (isinstance(data, dict) and (data.get("message") or data.get("error"))) or "Failed to append checklist to Notion page."
        raise HTTPException(status_code=r.status_code, detail=msg)
    return {"ok": True, "mode": "merge", "pageId": page_id, "added": added, "skipped": skipped}

# -----------------------
# OpenAI analyse (moved from Next.js /api/analyse)
# -----------------------
class AnalyseIn(BaseModel):
    log: str

class ResultSentiment(BaseModel):
    label: Literal["Positive", "Neutral", "Negative"]
    confidence: float

class ResultOut(BaseModel):
    summary: str
    blockers: List[str] = []
    nextSteps: List[str] = []
    sentiment: ResultSentiment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

@app.post("/analyse", response_model=ResultOut)
async def analyse(payload: AnalyseIn):
    log = (payload.log or "").strip()
    if not log:
        raise HTTPException(status_code=400, detail="Missing 'log' text.")

    # Fallback (no key): return deterministic sample so UI still works
    if not OPENAI_API_KEY:
        return ResultOut(
            summary=(
                "Fixed auth bug, added unit tests, paired on filters, investigated "
                "flaky CI and a pending sandbox API key."
            ),
            blockers=["Payments sandbox API key pending", "CI intermittent failures"],
            nextSteps=[
                "Add visual tests",
                "Wire error tracking",
                "Follow up with Ops for API key",
                "Investigate mobile pagination",
            ],
            sentiment=ResultSentiment(label="Neutral", confidence=0.7),
        )

    # Build JSON schema to enforce the exact shape
    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "SprintAnalysis",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "blockers": {"type": "array", "items": {"type": "string"}},
                    "nextSteps": {"type": "array", "items": {"type": "string"}},
                    "sentiment": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "required": ["summary", "blockers", "nextSteps", "sentiment"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    system_prompt = (
        "You are a sprint assistant. Given a developer's free-form day log, produce:\n"
        "- \"summary\": multiple bullet points in the way of meeting minutes type formatting.\n"
        "- \"blockers\": list of current blockers.\n"
        "- \"nextSteps\": concrete next actions as short imperatives.\n"
        "- \"sentiment\": overall mood label (Positive/Neutral/Negative) with confidence 0..1.\n"
        "Return only valid JSON that matches the schema."
    )

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Day log:\n{log}"},
        ],
        "response_format": json_schema,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.status_code >= 400:
            msg = None
            if isinstance(data, dict):
                msg = data.get("error") or data.get("message")
            raise HTTPException(status_code=502, detail=msg or "OpenAI error")

    raw = (data.get("choices", [{}])[0].get("message", {}).get("content") or "{}").strip()
    try:
        parsed = ResultOut.model_validate_json(raw)
    except Exception:
        # Be forgiving if provider returns slightly off JSON
        import json
        try:
            parsed_obj = json.loads(raw)
        except Exception:
            raise HTTPException(status_code=502, detail="Model returned non-JSON content")
        try:
            parsed = ResultOut.model_validate(parsed_obj)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Model JSON failed validation: {e}")

    return parsed

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
        r = client.get(url, headers={"Accept": "application/json"})
        try:
            data = r.json()
        except Exception:
            data = None
    if r.status_code != 200:
        detail = None
        if isinstance(data, dict):
            detail = data.get("message") or data.get("error")
        if not detail:
            # Fall back to raw text (trim to keep it readable)
            try:
                detail = (r.text or "").strip()[:200] or "Trello auth failed"
            except Exception:
                detail = "Trello auth failed"
        raise HTTPException(status_code=r.status_code, detail=detail)
    return {"ok": True, "me": {k: (data or {}).get(k) for k in ("id", "username", "fullName")}}

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
        r = client.post(url, params=params, headers={"Accept": "application/json"})
        try:
            data = r.json()
        except Exception:
            data = None
    if r.status_code != 200:
        msg = None
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error")
        if not msg:
            try:
                msg = (r.text or "").strip()[:200]
            except Exception:
                msg = None
        if not msg:
            msg = f"Trello error {r.status_code}"
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
        rl = client.get(url_lists, params=params, headers={"Accept": "application/json"})
        try:
            list_data = rl.json()
        except Exception:
            list_data = None
        if rl.status_code != 200:
            detail = None
            if isinstance(list_data, dict):
                detail = list_data.get("message") or list_data.get("error")
            if not detail:
                try:
                    detail = (rl.text or "").strip()[:200]
                except Exception:
                    detail = None
            raise HTTPException(status_code=rl.status_code, detail=detail or "Failed to read card checklists.")
        # Assume at most 1 checklist; create if none
        checklist = list_data[0] if isinstance(list_data, list) and list_data else None
        if not checklist:
            url_create = "https://api.trello.com/1/checklists"
            rc = client.post(
                url_create,
                params={"key": key, "token": tok, "idCard": card_id, "name": payload.checklistName or "Checklist"},
                headers={"Accept": "application/json"},
            )
            try:
                create_data = rc.json()
            except Exception:
                create_data = None
            if rc.status_code != 200:
                detail = None
                if isinstance(create_data, dict):
                    detail = create_data.get("message") or create_data.get("error")
                if not detail:
                    try:
                        detail = (rc.text or "").strip()[:200]
                    except Exception:
                        detail = None
                raise HTTPException(status_code=rc.status_code, detail=detail or "Failed to create checklist.")
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
        "TRELLO_TOKEN_set": bool(os.getenv("TRELLO_TOKEN")),
        "NOTION_API_KEY_set": bool(os.getenv("NOTION_API_KEY")),
        "NOTION_SUMMARY_PAGE_set": bool(os.getenv("NOTION_SUMMARY_PAGE")),
        "NOTION_CHECKLIST_PAGE_set": bool(os.getenv("NOTION_CHECKLIST_PAGE")),
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