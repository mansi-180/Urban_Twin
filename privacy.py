"""DSP layer: pseudonymization, encryption at rest, differential privacy, k-anonymity, role-based access."""
import hashlib, os, numpy as np
from cryptography.fernet import Fernet
from config import *
SALT = "city-twin-demo-salt"   # in production: environment variable / secrets manager

def pseudonymize(x): return hashlib.sha256((SALT + str(x)).encode()).hexdigest()[:10]
def hash_pw(p): return hashlib.sha256((SALT + p).encode()).hexdigest()

def ensure_key():
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(KEY):
        with open(KEY, "wb") as f: f.write(Fernet.generate_key())

def _fernet():
    with open(KEY, "rb") as f: return Fernet(f.read())

def encrypt_file(src, dst):
    with open(src, "rb") as f: data = f.read()
    with open(dst, "wb") as f: f.write(_fernet().encrypt(data))

def decrypt_bytes(path):
    with open(path, "rb") as f: return _fernet().decrypt(f.read())

def dp_noise(v, eps=1.0, sens=50):
    """Laplace mechanism: noise scale = sensitivity / epsilon (smaller eps = more privacy)."""
    v = np.asarray(v, dtype=float)
    return np.maximum(0, v + np.random.laplace(0, sens / eps, size=v.shape))

def k_anonymity(df, cols): return int(df.groupby(cols).size().min())

USERS = {"planner": (hash_pw("plan123"), "planner"), "public": (hash_pw("pub123"), "public")}
def login(u, p):
    r = USERS.get(u)
    return r[1] if r and r[0] == hash_pw(p) else None
