"""
Voice-Controlled Local AI Agent
Main Gradio UI application

Run with: python app.py
"""

import os
import gradio as gr
from dotenv import load_dotenv
from stt import transcribe_audio
from intent import classify_intent, format_intents_display
from agent import execute_intents, execute_confirmed_intents, format_results_markdown
from tools.file_ops import list_output_files

load_dotenv()

# ── Session state keys ──────────────────────────────────────────────────────
# We use gr.State to persist across interactions within a session

INTRO_MD = """
# 🎙️ Voice-Controlled Local AI Agent

Speak or upload audio to control your local AI agent. It will:
- **Transcribe** your speech using Whisper
- **Classify** your intent using a local LLM (Ollama)
- **Execute** the action — create files, write code, summarize text, or chat
- **Confirm** before any file writes (human-in-the-loop)

**Supported commands (examples):**
- *"Write a Python retry decorator and save it to retry.py"*
- *"Create a new file called notes.txt"*
- *"Summarize this: Machine learning is a branch of AI..."*
- *"What is the capital of France?"*
- *"Summarize this text and save it to summary.txt: ..."* (compound!)
"""


def process_audio(audio, auto_confirm, chat_history_state, pending_state):
    """Main pipeline: audio → STT → intent → execute → display."""
    if audio is None:
        return (
            "⚠️ No audio provided. Please record or upload audio.",
            "_Waiting..._",
            "_Waiting..._",
            chat_history_state,
            pending_state,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # ── Step 1: Transcribe ──────────────────────────────────────────────────
    try:
        transcript = transcribe_audio(audio)
    except Exception as e:
        return (
            f"❌ Transcription failed: {str(e)}",
            "_Failed_", "_Failed_",
            chat_history_state, pending_state,
            gr.update(visible=False), gr.update(visible=False),
        )

    if not transcript:
        return (
            "⚠️ Could not understand audio. Please speak clearly and try again.",
            "_No speech detected_", "_No output_",
            chat_history_state, pending_state,
            gr.update(visible=False), gr.update(visible=False),
        )

    # ── Step 2: Classify intent ─────────────────────────────────────────────
    try:
        intents = classify_intent(transcript)
        intent_display = format_intents_display(intents)
    except Exception as e:
        intent_display = f"❌ Intent classification failed: {str(e)}"
        intents = [{"intent": "general_chat", "params": {"description": transcript}}]

    # ── Step 3: Execute ─────────────────────────────────────────────────────
    confirm_needed = not auto_confirm

    try:
        results, pending = execute_intents(
            intents,
            confirm_file_ops=confirm_needed,
            chat_history=chat_history_state,
        )
    except Exception as e:
        return (
            transcript, intent_display,
            f"❌ Execution error: {str(e)}",
            chat_history_state, pending_state,
            gr.update(visible=False), gr.update(visible=False),
        )

    # Update chat history for general_chat responses
    new_history = list(chat_history_state) if chat_history_state else []
    for r in results:
        if r.get("intent") == "general_chat":
            new_history.append({"role": "user", "content": transcript})
            new_history.append({"role": "assistant", "content": r.get("output", "")})

    output_md = format_results_markdown(results)

    # ── Step 4: Handle confirmation needed ─────────────────────────────────
    if pending:
        pending_summary = "\n".join(
            f"- **{p['intent']}**: `{p['params'].get('filename', 'file')}`"
            for p in pending
        )
        confirmation_text = (
            f"⚠️ **Confirm file operations:**\n\n{pending_summary}\n\n"
            "These files will be written to the `output/` folder."
        )
        if output_md and output_md != "_No results yet._":
            output_md = output_md + "\n\n---\n\n" + confirmation_text
        else:
            output_md = confirmation_text

        return (
            transcript, intent_display, output_md,
            new_history, pending,
            gr.update(visible=True, value="✅ Yes, proceed"),
            gr.update(visible=True, value="❌ Cancel"),
        )

    return (
        transcript, intent_display, output_md,
        new_history, [],
        gr.update(visible=False),
        gr.update(visible=False),
    )


def confirm_execution(pending_state, chat_history_state, current_output):
    """Execute the pending file operations after user confirmation."""
    if not pending_state:
        return current_output, [], gr.update(visible=False), gr.update(visible=False)

    try:
        results = execute_confirmed_intents(pending_state)
        new_output = format_results_markdown(results)
        combined = (current_output or "") + "\n\n---\n\n✅ **Confirmed — executing...**\n\n" + new_output
    except Exception as e:
        combined = (current_output or "") + f"\n\n❌ Execution error: {str(e)}"

    return (
        combined, [],
        gr.update(visible=False),
        gr.update(visible=False),
    )


def cancel_execution(current_output):
    return (
        (current_output or "") + "\n\n🚫 **File operation cancelled.**",
        [],
        gr.update(visible=False),
        gr.update(visible=False),
    )


def refresh_files():
    files = list_output_files()
    if not files:
        return "_No files yet. Execute a command to create some._"
    rows = []
    for f in files:
        icon = "📄" if f["type"] == "file" else "📁"
        size = f"{f['size']} bytes" if f["type"] == "file" else ""
        rows.append(f"| {icon} `{f['name']}` | {size} |")
    header = "| File | Size |\n|------|------|\n"
    return header + "\n".join(rows)


def clear_all(chat_history_state):
    return "", "_Waiting..._", "_Waiting..._", [], [], gr.update(visible=False), gr.update(visible=False)


# ── Build UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Voice AI Agent") as demo:

    # ── State ───────────────────────────────────────────────────────────────
    chat_history = gr.State([])
    pending_intents = gr.State([])

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown(INTRO_MD)

    with gr.Row():
        # ── Left column: Input ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Audio Input")

            with gr.Tabs():
                with gr.TabItem("🎙️ Record"):
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Click to record",
                    )
                with gr.TabItem("📂 Upload"):
                    file_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload .wav or .mp3",
                    )

            auto_confirm = gr.Checkbox(
                label="⚡ Auto-confirm file operations (skip confirmation step)",
                value=False,
            )

            with gr.Row():
                submit_mic = gr.Button("🚀 Process Recording", variant="primary")
                submit_file = gr.Button("🚀 Process Upload", variant="primary")

            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        # ── Right column: Pipeline results ──────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Pipeline Results")

            with gr.Group():
                transcript_box = gr.Textbox(
                    label="📝 Transcribed Text",
                    placeholder="Your speech will appear here...",
                    lines=2,
                    interactive=False,
                    elem_classes=["transcript-box"],
                )

                intent_box = gr.Markdown(
                    value="_Waiting..._",
                    label="🎯 Detected Intent(s)",
                )

                output_box = gr.Markdown(
                    value="_Waiting..._",
                    label="⚙️ Action & Output",
                    elem_classes=["output-box"],
                )

            # Confirmation buttons (hidden by default)
            with gr.Row(elem_classes=["confirm-row"]):
                confirm_btn = gr.Button("✅ Yes, proceed", variant="primary", visible=False)
                cancel_btn = gr.Button("❌ Cancel", variant="stop", visible=False)

    # ── Output files panel ──────────────────────────────────────────────────
    with gr.Accordion("📁 Output Files", open=False):
        files_display = gr.Markdown("_No files yet._")
        refresh_btn = gr.Button("🔄 Refresh")

    # ── Session history ──────────────────────────────────────────────────────
    with gr.Accordion("💬 Chat History", open=False):
        history_display = gr.Chatbot(label="Conversation", height=300)

    # ── Event wiring ─────────────────────────────────────────────────────────

    shared_outputs = [
        transcript_box, intent_box, output_box,
        chat_history, pending_intents,
        confirm_btn, cancel_btn,
    ]

    submit_mic.click(
        fn=process_audio,
        inputs=[mic_input, auto_confirm, chat_history, pending_intents],
        outputs=shared_outputs,
    )

    submit_file.click(
        fn=process_audio,
        inputs=[file_input, auto_confirm, chat_history, pending_intents],
        outputs=shared_outputs,
    )

    confirm_btn.click(
        fn=confirm_execution,
        inputs=[pending_intents, chat_history, output_box],
        outputs=[output_box, pending_intents, confirm_btn, cancel_btn],
    )

    cancel_btn.click(
        fn=cancel_execution,
        inputs=[output_box],
        outputs=[output_box, pending_intents, confirm_btn, cancel_btn],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[chat_history],
        outputs=shared_outputs,
    )

    refresh_btn.click(fn=refresh_files, outputs=[files_display])

    # Sync chatbot display with history state
    def update_chatbot(history):
        pairs = []
        for i in range(0, len(history) - 1, 2):
            if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
                pairs.append((history[i]["content"], history[i+1]["content"]))
        return pairs

    chat_history.change(fn=update_chatbot, inputs=[chat_history], outputs=[history_display])


if __name__ == "__main__":
    print("🎙️ Starting Voice AI Agent...")
    print(f"📂 Output folder: {os.path.abspath('output/')}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate",
        ),
    )