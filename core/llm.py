import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"


def get_groq_llm(temperature: float = 0.3) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=temperature,
    )


def invoke_llm_chain(chain: Any, input_value: Any) -> Any:
    try:
        return chain.invoke(input_value)
    except Exception as exc:
        error_text = str(exc).lower()
        if "model_not_found" in error_text or "model not found" in error_text:
            raise RuntimeError(
                f"Groq model is invalid or unavailable ({GROQ_MODEL}): {exc}"
            ) from exc
        if "429" in error_text or "rate limit" in error_text or "resource_exhausted" in error_text:
            raise RuntimeError(
                f"Groq rate limit reached (HTTP 429): {exc}"
            ) from exc
        if "401" in error_text or "403" in error_text or "authentication" in error_text or "permission" in error_text:
            raise RuntimeError(
                f"Groq authentication failed. Check GROQ_API_KEY in your .env file: {exc}"
            ) from exc
        if any(term in error_text for term in ("timeout", "connection", "network", "unavailable")):
            raise RuntimeError(
                f"Groq network error. Check your connection and try again: {exc}"
            ) from exc
        raise RuntimeError(f"Groq API request failed: {exc}") from exc
