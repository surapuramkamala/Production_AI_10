
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import JWTError, jwt
from pwdlib import PasswordHash

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "")
if len(JWT_SECRET) < 32:
    raise RuntimeError(
        "Set JWT_SECRET to a random secret of at least 32 characters."
    )

JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = 60 * 8
password_hash = PasswordHash.recommended()
DB_PATH = os.getenv("AUTH_DB_PATH", "users.db")




def init_auth_db():
    db_directory = os.path.dirname(DB_PATH)
    if db_directory:
        os.makedirs(db_directory, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)



def create_user(email: str, password: str):
    email = email.strip().lower()

    if "@" not in email or len(email) > 254:
        raise ValueError("Enter a valid email address.")

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    hashed = password_hash.hash(password)

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute(
                """INSERT INTO users (email, password_hash, created_at)
                   VALUES (?, ?, ?)""",
                (
                    email,
                    hashed,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")

    return {"id": user_id, "email": email}


def verify_user(email: str, password: str):
    email = email.strip().lower()

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not row or not password_hash.verify(password, row[2]):
        return None

    return {"id": row[0], "email": row[1]}


def create_access_token(user: dict) -> str:
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_MINUTES
    )

    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "exp": expires,
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str):
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
    except JWTError:
        return None
