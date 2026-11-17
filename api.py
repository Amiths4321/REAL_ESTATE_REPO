import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from app import setup_database

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import HISTORY_LIMIT, new_memory, run_agent
from fastapi import Depends



# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: if every request comes from one Streamlit server, they all share
# one IP address, so this limit applies to all users together.
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))


# ==================================================
# CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Real Estate AI API",
    description="AI-powered Mumbai property assistant",
    version="1.0.0"
)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

@app.on_event("startup")
def initialize_database():
    setup_database()
    print("Database initialized successfully.", flush=True)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

API_SECRET_KEY = os.getenv("API_SECRET_KEY")

def verify_api_key(
    x_api_key: str = Header(...)
):
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="API authentication is not configured."
        )

    if not secrets.compare_digest(
        x_api_key,
        API_SECRET_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )
# ==================================================
# ALLOW LOCAL FRONTEND CONNECTIONS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==================================================
# DATA MODELS
# ==================================================

class ChatRequest(BaseModel):

    session_id: str = Field(
        min_length=1,
        max_length=100
    )

    question: str = Field(
        min_length=2,
        max_length=1000
    )

    @field_validator("session_id", "question")
    @classmethod
    def remove_extra_spaces(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty.")

        return value


class ChatResponse(BaseModel):

    session_id: str

    question: str

    answer: str

    memory: dict


# ==================================================
# TEMPORARY SESSION STORAGE (in memory, lost on restart)
# ==================================================

sessions = OrderedDict()
sessions_lock = threading.Lock()


def get_session(session_id):
    """Return the session for this ID, creating it if needed.

    The oldest sessions are dropped once MAX_SESSIONS is exceeded,
    so memory use cannot grow without limit.
    """

    with sessions_lock:

        session = sessions.get(session_id)

        if session is None:

            session = {
                "history": [],
                "memory": new_memory(),
                # Stops two requests on one session running at once
                "lock": threading.Lock()
            }

            sessions[session_id] = session

            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)

        else:
            sessions.move_to_end(session_id)

        return session


# ==================================================
# HEALTH CHECK ENDPOINT
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "running",
        "application": "Real Estate AI Assistant"
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post("/chat")
@limiter.limit("10/minute")

def chat(
    request: Request,
    body: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    # slowapi needs the `request: Request` parameter, so the
    # validated JSON body is called `body`.

    session_id = body.session_id
    question = body.question

    session = get_session(session_id)

    try:

        with session["lock"]:

            start_time = time.perf_counter()

            answer = run_agent(
                question,
                session["history"],
                session["memory"]
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "AI request completed in %.2f seconds",
                elapsed
            )

            # Update conversation history
            session["history"].append({
                "role": "user",
                "content": question
            })

            session["history"].append({
                "role": "assistant",
                "content": answer
            })

            # Retain only the most recent messages
            session["history"] = session["history"][-HISTORY_LIMIT:]

            memory_snapshot = dict(session["memory"])

        return ChatResponse(
            session_id=session_id,
            question=question,
            answer=answer,
            memory=memory_snapshot
        )

    except Exception as e:
        import traceback

        print("CHAT API ERROR:", repr(e), flush=True)
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )