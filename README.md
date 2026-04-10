# 🎙️ Voice-Controlled Local AI Agent

A fully local voice agent that transcribes speech, classifies intent, and executes actions on your machine — all through a clean web UI.

## 📸 Demo

> Record or upload audio → the agent transcribes it, detects your intent, and acts: creating files, writing code, summarizing text, or chatting.

## 🏗️ Architecture

```
Audio Input (mic / file)
        │
        ▼
┌─────────────────┐
│  Speech-to-Text  │  Whisper (HuggingFace local) OR Groq API
└────────┬────────┘
         │  transcript
         ▼
┌─────────────────────┐
│  Intent Classifier   │  Local LLM via Ollama (llama3.2 / mistral)
└────────┬────────────┘
         │  JSON intents list
         ▼
┌──────────────────────────────────┐
│         Agent Orchestrator        │  Compound command support
│  ┌──────────┐  ┌──────────────┐  │  Human-in-the-loop confirmation
│  │ File ops │  │  Text ops    │  │
│  │ create   │  │  write_code  │  │
│  │ folder   │  │  summarize   │  │
│  └──────────┘  │  general_chat│  │
│                └──────────────┘  │
└─────────────────┬────────────────┘
                  │  results
                  ▼
         ┌─────────────┐
         │  Gradio UI   │  Transcript · intents · action · output
         └─────────────┘
                  │
                  ▼
            output/  (sandboxed)
```

## ✨ Features

- **🎤 Dual audio input** — live microphone recording or file upload (.wav, .mp3)
- **📝 Local STT** — Whisper `base` model via HuggingFace (CUDA if available, CPU fallback)
- **🧠 LLM intent classification** — Ollama with structured JSON output, keyword fallback
- **⚡ Compound commands** — "Summarize this AND save it to summary.txt" works
- **🔒 Human-in-the-loop** — optional confirmation before any file is written
- **🛡️ Sandboxed output** — ALL file writes go to `output/` only, no path traversal possible
- **💬 Session memory** — chat history maintained within a session
- **📁 Output file browser** — see all generated files in the UI
- **🌙 Graceful degradation** — keyword fallback if Ollama is down, Groq fallback if local Whisper is slow

## 🚀 Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 1. Clone & install

```bash
git clone https://github.com/yourusername/voice-agent.git
cd voice-agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2       # recommended (fast, 2GB)
# or
ollama pull mistral        # alternative
ollama serve               # start Ollama server
```

### 4. Run the agent

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

## 🗣️ Example Commands

| What you say | What happens |
|---|---|
| "Write a Python retry decorator and save it to retry.py" | Generates code, saves to `output/retry.py` |
| "Create a new file called notes.txt" | Creates empty `output/notes.txt` |
| "Summarize this: Machine learning is a subfield..." | Returns a bullet-point summary |
| "Summarize this text and save it to summary.txt: ..." | Compound: summarizes AND saves the file |
| "What's the difference between TCP and UDP?" | General chat response |

## 🔧 Configuration

All settings live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `STT_BACKEND` | `local` | `local` = HuggingFace Whisper, `groq` = Groq API |
| `GROQ_API_KEY` | — | Required only if `STT_BACKEND=groq` |
| `OLLAMA_MODEL` | `llama3.2` | Any model you've pulled with Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

## ⚙️ Hardware Notes

### STT

The default `openai/whisper-base` model (~74M params) runs fine on CPU (2–5s per clip). For faster transcription:

- **GPU (CUDA)**: automatically detected and used — transcription drops to <1s
- **Slow machine**: set `STT_BACKEND=groq` in `.env` and add your [Groq API key](https://console.groq.com) (free tier, uses `whisper-large-v3`)

### LLM (Intent + Code Gen)

Ollama runs the LLM locally. Recommended models by hardware:

| RAM | Model |
|---|---|
| 4GB | `llama3.2:1b` or `phi3:mini` |
| 8GB | `llama3.2` (default) or `mistral` |
| 16GB+ | `llama3.1:8b` or `mixtral` |

## 📁 Output Directory

All generated files are sandboxed inside `output/`. The agent enforces this with path sanitization — it's impossible to write outside this folder via voice commands.

## 🧪 Running Tests

```bash
# Test intent classification without audio
python -c "
from intent import classify_intent
print(classify_intent('Create a Python file with a bubble sort algorithm'))
"

# Test STT with a file
python -c "
from stt import transcribe_audio
print(transcribe_audio('test.wav'))
"
```

## 📝 Bonus Features Implemented

- [x] **Compound commands** — multiple intents per audio clip
- [x] **Human-in-the-loop** — toggle-able confirmation before file writes
- [x] **Graceful degradation** — keyword fallback + Groq STT fallback
- [x] **Session memory** — chat context maintained within session
- [ ] Model benchmarking (see article)

## 📄 Article

> Architecture deep-dive, model comparisons, and challenges: [Link to article]

## 📬 Submission

Submitted via [https://forms.gle/5x32P7zr4NvyRgK6A](https://forms.gle/5x32P7zr4NvyRgK6A)
