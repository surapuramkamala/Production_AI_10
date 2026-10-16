
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from auth import (
    create_access_token,
    create_user,
    decode_access_token,
    init_auth_db,
    verify_user,
)
from rag_engine import EnterpriseRAG


rag = EnterpriseRAG()
bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_db()
    yield


app = FastAPI(
    title="Enterprise RAG Application",
    version="1.0.0",
    lifespan=lifespan,
)


class AuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    stage: str = "hybrid"


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in to continue.",
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired login token.",
        )

    return {
        "id": payload["sub"],
        "email": payload.get("email", ""),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "nvidia_configured": True,
        "indexed_chunks": len(rag.chunks),
        "index_loaded": rag.index is not None,
    }


@app.post("/api/auth/register")
def register(request: AuthRequest):
    try:
        user = create_user(request.email, request.password)
        token = create_access_token(user)
        return {"message": "Account created.", "access_token": token,
                "user": user}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/auth/login")
def login(request: AuthRequest):
    user = verify_user(request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
        )

    return {
        "message": "Login successful.",
        "access_token": create_access_token(user),
        "user": {"email": user["email"]},
    }


@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return {"user": user}


@app.post("/api/ask")
def ask(
    request: AskRequest,
    user=Depends(get_current_user),
):
    allowed_stages = {
        "basic", "hybrid", "reranking", "query_transformation"
    }

    if request.stage not in allowed_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Choose one of: {', '.join(sorted(allowed_stages))}",
        )

    try:
        return rag.answer_question(
            question=request.question,
            stage=request.stage,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Log the full exception on the server in production.
        raise HTTPException(
            status_code=502,
            detail=f"Could not complete the AI request: {exc}",
        )


@app.post("/api/admin/reindex")
def reindex(user=Depends(get_current_user)):
    try:
        return {
            "message": "Index rebuilt successfully.",
            **rag.index_documents(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/vector-store")
def vector_store(user=Depends(get_current_user)):
    return {
        "indexed_chunks": len(rag.chunks),
        "index_loaded": rag.index is not None,
    }


# Mount the frontend last so it does not intercept API routes.
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount(
    "/",
    StaticFiles(directory=str(static_dir), html=True),
    name="frontend",
)
