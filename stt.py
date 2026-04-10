"""
Speech-to-Text module using OpenAI Whisper via HuggingFace transformers.
Falls back to Groq API if local model is unavailable or too slow.
"""

import os
import tempfile
import numpy as np

STT_BACKEND = os.getenv("STT_BACKEND", "local")  # "local" or "groq"


def transcribe_audio(audio_input) -> str:
    """
    Transcribe audio to text.
    audio_input: can be a file path (str) or a tuple (sample_rate, np.ndarray) from Gradio mic
    """
    if STT_BACKEND == "groq":
        return _transcribe_groq(audio_input)
    else:
        return _transcribe_local(audio_input)


def _transcribe_local(audio_input) -> str:
    """Use Whisper locally via HuggingFace transformers pipeline."""
    try:
        from transformers import pipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[STT] Loading Whisper on {device}...")

        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device,
            chunk_length_s=30,
        )

        audio_path = _prepare_audio(audio_input)
        result = asr(audio_path)
        text = result["text"].strip()
        print(f"[STT] Transcribed: {text}")
        return text

    except Exception as e:
        print(f"[STT] Local transcription failed: {e}")
        print("[STT] Falling back to Groq API...")
        return _transcribe_groq(audio_input)


def _transcribe_groq(audio_input) -> str:
    """Use Groq's Whisper API for transcription."""
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

        client = Groq(api_key=api_key)
        audio_path = _prepare_audio(audio_input)

        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text",
            )

        text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        print(f"[STT] Groq transcribed: {text}")
        return text

    except Exception as e:
        raise RuntimeError(f"Groq transcription failed: {e}")


def _prepare_audio(audio_input) -> str:
    """
    Normalizes audio_input to a file path.
    Handles: file path strings, (sample_rate, ndarray) tuples from Gradio mic.
    """
    import soundfile as sf

    if isinstance(audio_input, str):
        return audio_input

    if isinstance(audio_input, tuple):
        sample_rate, audio_array = audio_input
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_array, sample_rate)
        return tmp.name

    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
