"""
Agent orchestrator — receives classified intents and routes them to the correct tools.
Handles single and compound commands, confirmation prompts, and error recovery.
"""

from typing import List, Tuple
from tools.file_ops import create_file, create_folder
from tools.text_ops import generate_code, summarize_text, general_chat


def execute_intents(
    intents: List[dict],
    confirm_file_ops: bool = True,
    chat_history: list = None,
) -> Tuple[List[dict], bool]:
    """
    Execute a list of intents sequentially.

    Args:
        intents: list of intent dicts from classify_intent()
        confirm_file_ops: if True, return needs_confirmation=True for file writes
        chat_history: running chat history for context

    Returns:
        (results, needs_confirmation)
        - results: list of result dicts
        - needs_confirmation: True if a file-write intent is pending confirmation
    """
    results = []
    pending_file_ops = []

    # Split intents: collect file ops for confirmation, execute others immediately
    for item in intents:
        intent = item.get("intent")
        if intent in ("create_file", "write_code") and confirm_file_ops:
            pending_file_ops.append(item)
        else:
            result = _execute_single(item, chat_history)
            results.append(result)

    if pending_file_ops and confirm_file_ops:
        # Return early — UI will ask user to confirm
        return results, pending_file_ops

    return results, []


def execute_confirmed_intents(intents: List[dict], generated_contents: dict = None) -> List[dict]:
    """Execute file-writing intents after user confirmation."""
    results = []
    for item in intents:
        result = _execute_single(item, generated_contents=generated_contents)
        results.append(result)
    return results


def _execute_single(item: dict, chat_history: list = None, generated_contents: dict = None) -> dict:
    """Execute a single intent and return a result dict."""
    intent = item.get("intent", "general_chat")
    params = item.get("params", {})
    description = params.get("description", "")
    filename = params.get("filename", "output.txt")
    language = params.get("language", "python")

    try:
        if intent == "write_code":
            # Generate code
            code = generate_code(description, language)
            # Save to file
            result = create_file(filename, code)
            result["intent"] = "write_code"
            result["action"] = f"Generated {language} code → saved as `{result.get('filename', filename)}`"
            result["output"] = code
            return result

        elif intent == "create_file":
            # If we have pre-generated content (e.g. from a summarize step), use it
            content = ""
            if generated_contents and filename in generated_contents:
                content = generated_contents[filename]
            result = create_file(filename, content)
            result["intent"] = "create_file"
            result["action"] = f"Created file `{result.get('filename', filename)}`"
            result["output"] = content or "(empty file)"
            return result

        elif intent == "create_folder":
            result = create_folder(params.get("folder_name", "new_folder"))
            result["intent"] = "create_folder"
            result["action"] = f"Created folder `{result.get('filename', 'folder')}`"
            result["output"] = ""
            return result

        elif intent == "summarize":
            text_to_summarize = params.get("text_to_summarize", description)
            summary = summarize_text(text_to_summarize)
            return {
                "status": "success",
                "intent": "summarize",
                "action": "Summarized text",
                "output": summary,
                "filename": None,
            }

        elif intent == "general_chat":
            response = general_chat(description, chat_history)
            return {
                "status": "success",
                "intent": "general_chat",
                "action": "Responded to query",
                "output": response,
                "filename": None,
            }

        else:
            return {
                "status": "error",
                "intent": intent,
                "action": "Unknown intent",
                "output": f"I don't know how to handle intent: `{intent}`",
                "filename": None,
            }

    except Exception as e:
        return {
            "status": "error",
            "intent": intent,
            "action": "Execution failed",
            "output": f"Error: {str(e)}",
            "filename": None,
        }


def format_results_markdown(results: List[dict]) -> str:
    """Format execution results for display in the UI."""
    if not results:
        return "_No results yet._"

    parts = []
    for r in results:
        intent = r.get("intent", "unknown")
        action = r.get("action", "")
        output = r.get("output", "")
        status = r.get("status", "success")
        filename = r.get("filename")

        icon = {
            "write_code": "✍️",
            "create_file": "📄",
            "create_folder": "📁",
            "summarize": "📝",
            "general_chat": "💬",
        }.get(intent, "⚙️")

        status_icon = "✅" if status == "success" else "❌"

        section = f"{status_icon} {icon} **{action}**\n"
        if filename:
            section += f"📂 Saved to: `output/{filename}`\n"
        section += f"\n{output}"
        parts.append(section)

    return "\n\n---\n\n".join(parts)
