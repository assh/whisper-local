from cryptography.fernet import Fernet
import base64, os

FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    # generate once: print(base64.urlsafe_b64encode(os.urandom(32)).decode())
    FERNET_KEY = "REPLACE_WITH_32BYTE_URLSAFE_BASE64"
fernet = Fernet(FERNET_KEY)

def enc(txt: str) -> str:
    return fernet.encrypt(txt.encode()).decode()

def dec(txt: str | None) -> str | None:
    if not txt: return None
    return fernet.decrypt(txt.encode()).decode()