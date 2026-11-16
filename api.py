
from fastapi import FastAPI, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import run_agent

import time
import logging

logger = logging.getLogger(__name__)

start_time = time.perf_counter()

answer = run_agent(question, history, memory)

elapsed = time.perf_counter() - start_time

logger.info(
    "AI request completed in %.2f seconds",
    elapsed
)

# CREATE FASTAPI APPLICATION

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

# ALLOW LOCAL FRONTEND CONNECTIONS

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


# TEMPORARY SESSION STORAGE

sessions = {}


# REQUEST DATA MODEL


from pydantic import BaseModel, Field, field_validator


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

@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    # Keep your existing chat endpoint logic here


# RESPONSE DATA MODEL

class ChatResponse(BaseModel):

    session_id: str

    question: str

    answer: str

    memory: dict


# CREATE DEFAULT MEMORY

def create_memory():

    return {
        "locations": [],
        "property_type": None,
        "bedrooms": None,
        "min_price": None,
        "max_price": None
    }


# HEALTH CHECK ENDPOINT

@app.get("/health")
def health_check():

    return {
        "status": "running",
        "application": "Real Estate AI Assistant"
    }


# CHAT ENDPOINT

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Create a session if it doesn't exist

    if session_id not in sessions:

        sessions[session_id] = {
            "history": [],
            "memory": create_memory()
        }

    session = sessions[session_id]

    history = session["history"]

    memory = session["memory"]

    try:

        # Run your existing AI agent

        answer = run_agent(
            question,
            history,
            memory
        )

        # Update conversation history

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        # Retain the last six messages

        session["history"] = history[-6:]

        return ChatResponse(
            session_id=session_id,
            question=question,
            answer=answer,
            memory=memory
        )

    except Exception as error:
        logger.exception("AI request failed")

        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed.Please try again"
        )
