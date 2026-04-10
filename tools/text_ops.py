"""
Text processing tools — code generation and summarization using Ollama LLM.
"""

import os
import requests

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_complete(system: str, user: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_code(description: str, language: str = "python") -> str:
    """
    Generate code based on a description.
    Returns only the code block (no explanation).
    """
    system = f"""You are an expert {language} developer.
Generate clean, well-commented {language} code based on the user's description.
Return ONLY the code — no explanations, no markdown fences, no preamble.
The code should be complete, runnable, and follow best practices."""

    user = f"Write {language} code for: {description}"
    return _ollama_complete(system, user)


def summarize_text(text: str) -> str:
    """
    Summarize the provided text.
    """
    system = """You are a concise summarizer. 
Summarize the provided text in clear, plain English.
Use bullet points for key facts. Keep it under 150 words."""

    user = f"Summarize this:\n\n{text}"
    return _ollama_complete(system, user)


def general_chat(message: str, history: list = None) -> str:
    """
    Handle general conversation.
    history: list of {"role": ..., "content": ...} dicts
    """
    system = """You are a helpful, friendly AI assistant integrated into a voice agent.
Be concise and clear since your responses will be read on screen."""

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])  # Keep last 3 exchanges for context
    messages.append({"role": "user", "content": message})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
