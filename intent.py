"""
Intent classification module using a local LLM via Ollama.
Falls back to simple keyword matching if Ollama is unavailable.
"""

import json
import re
import os
from typing import List

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.
Analyze the user's transcribed speech and extract ALL intents present.

Respond ONLY with a valid JSON array (no markdown, no explanation). Each intent object:
{
  "intent": one of ["create_file", "write_code", "summarize", "general_chat"],
  "params": {
    "filename": "optional suggested filename",
    "language": "optional programming language for write_code",
    "description": "what the user wants",
    "text_to_summarize": "the text to summarize if intent is summarize"
  }
}

Rules:
- Multiple intents are allowed (compound commands)
- For write_code, always include a filename (e.g. "retry.py")
- For create_file, include the filename and optional initial content
- For summarize, extract the text to summarize from the command
- If nothing specific matches, use general_chat

Examples:
"Create a Python file with a retry function" →
[{"intent":"write_code","params":{"filename":"retry.py","language":"python","description":"a retry decorator/function"}}]

"Summarize this text and save it to summary.txt: The quick brown fox..." →
[{"intent":"summarize","params":{"description":"summarize the text","text_to_summarize":"The quick brown fox..."}},
 {"intent":"create_file","params":{"filename":"summary.txt","description":"save the summary"}}]
"""


def classify_intent(text: str) -> List[dict]:
    """
    Classify the intent(s) from transcribed text.
    Returns a list of intent dicts, each with 'intent' and 'params' keys.
    """
    if not text or not text.strip():
        return [{"intent": "general_chat", "params": {"description": ""}}]

    try:
        return _classify_ollama(text)
    except Exception as e:
        print(f"[Intent] Ollama failed: {e}. Using keyword fallback.")
        return _classify_keywords(text)


def _classify_ollama(text: str) -> List[dict]:
    """Use local Ollama LLM for intent classification."""
    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": f'Classify this command: "{text}"'},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    intents = json.loads(raw)
    if isinstance(intents, dict):
        intents = [intents]

    print(f"[Intent] Classified: {intents}")
    return intents


def _classify_keywords(text: str) -> List[dict]:
    """Simple keyword-based fallback classifier."""
    text_lower = text.lower()
    intents = []

    code_keywords = ["write code", "create code", "generate code", "write a", "create a function",
                     "write function", "python file", "javascript file", "code file", ".py", ".js", ".ts"]
    file_keywords = ["create file", "make file", "new file", "create folder", "make folder"]
    summarize_keywords = ["summarize", "summary", "sum up", "tldr", "brief me"]

    # Check for code writing
    if any(k in text_lower for k in code_keywords):
        lang = "python"
        if "javascript" in text_lower or ".js" in text_lower:
            lang = "javascript"
        elif "typescript" in text_lower:
            lang = "typescript"
        elif "bash" in text_lower or "shell" in text_lower:
            lang = "bash"

        ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "bash": "sh"}
        ext = ext_map.get(lang, "txt")
        intents.append({
            "intent": "write_code",
            "params": {
                "filename": f"output.{ext}",
                "language": lang,
                "description": text,
            }
        })

    # Check for file creation
    elif any(k in text_lower for k in file_keywords):
        intents.append({
            "intent": "create_file",
            "params": {
                "filename": "new_file.txt",
                "description": text,
            }
        })

    # Check for summarize
    if any(k in text_lower for k in summarize_keywords):
        intents.append({
            "intent": "summarize",
            "params": {
                "description": text,
                "text_to_summarize": text,
            }
        })

    if not intents:
        intents.append({
            "intent": "general_chat",
            "params": {"description": text}
        })

    return intents


def format_intents_display(intents: List[dict]) -> str:
    """Format intents for display in the UI."""
    lines = []
    for i, item in enumerate(intents, 1):
        intent = item.get("intent", "unknown")
        params = item.get("params", {})
        desc = params.get("description", "")
        filename = params.get("filename", "")

        label_map = {
            "write_code": "✍️ Write code",
            "create_file": "📄 Create file",
            "summarize": "📝 Summarize",
            "general_chat": "💬 General chat",
        }
        label = label_map.get(intent, intent)

        line = f"**{i}. {label}**"
        if filename:
            line += f" → `{filename}`"
        if desc:
            line += f"\n   _{desc[:100]}_"
        lines.append(line)

    return "\n".join(lines)
