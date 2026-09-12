"""FastAPI bridge: app.html -> HTTP -> existing main.py pipeline."""

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import run_pipeline
from core.rag_engine import ask_question

app = FastAPI(title="AI Video Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

_rag_chain: Any = None


class ProcessRequest(BaseModel):
    source: str
    language: str = "english"


class ChatRequest(BaseModel):
    question: str


def _language_for_pipeline(language: str) -> str:
    return "hinglish" if language.lower() in {"hi", "hinglish"} else "english"


def _result_payload(result: dict) -> dict:
    global _rag_chain
    _rag_chain = result.get("rag_chain")
    return {
        "success": True,
        "title": result.get("title", ""),
        "summary": result.get("summary", ""),
        "transcript": result.get("transcript", ""),
        "action_items": result.get("action_items", ""),
        "key_decisions": result.get("key_decisions", ""),
        "open_questions": result.get("open_questions", ""),
        "error": None,
    }


def _run(source: str, language: str) -> dict:
    if not source.strip():
        raise HTTPException(status_code=400, detail="A YouTube URL or local file is required.")
    try:
        return _result_payload(run_pipeline(source.strip(), _language_for_pipeline(language)))
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "model_not_found" in message or "model not found" in message:
            detail = str(exc)
        elif "groq_api_key" in message or "authentication" in message or "permission" in message:
            detail = "Groq authentication failed. Check GROQ_API_KEY in your .env file."
        elif "429" in message or "rate limit" in message or "resource_exhausted" in message:
            detail = "Groq rate limit reached (HTTP 429). Please wait and try again."
        elif "network" in message or "connection" in message or "timeout" in message:
            detail = "A network error occurred while processing the video. Please try again."
        else:
            detail = "Processing failed. Check the source and server logs for details."
        raise HTTPException(status_code=500, detail=detail) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process")
def process(request: ProcessRequest) -> dict:
    """Process a URL or a path visible to the server using the existing pipeline."""
    return _run(request.source, request.language)


@app.post("/process-upload")
def process_upload(
    file: UploadFile = File(...),
    language: str = Form("english"),
) -> dict:
    """Accept a browser upload, then pass its temporary path to the existing pipeline."""
    suffix = Path(file.filename or "upload").suffix
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(file.file.read())
        return _run(temporary_path, language)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A question is required.")
    if _rag_chain is None:
        raise HTTPException(status_code=409, detail="Process a video before asking questions.")
    try:
        return {"success": True, "answer": ask_question(_rag_chain, request.question.strip()), "error": None}
    except Exception as exc:
        message = str(exc).lower()
        if "model_not_found" in message or "model not found" in message:
            detail = str(exc)
        elif "429" in message or "rate limit" in message or "resource_exhausted" in message:
            detail = "Groq rate limit reached (HTTP 429). Please wait and try again."
        elif "authentication" in message or "permission" in message:
            detail = "Groq authentication failed. Check GROQ_API_KEY in your .env file."
        else:
            detail = "Unable to answer that question. Please try again."
        raise HTTPException(status_code=500, detail=detail) from exc
