"""
TEAM-33 AI Agent - Speech-to-Text (ASR) Module
Uses OpenAI Whisper for accurate speech recognition
"""

import os
import io
import logging
from typing import Dict, Tuple, Optional
import openai
from config import config

# Setup logging
logger = logging.getLogger(__name__)


class WhisperASR:
    """Speech-to-Text module using OpenAI Whisper"""

    def __init__(self):
        """Initialize Whisper ASR service"""
        self.api_key = config.OPENAI_API_KEY
        self.model = config.WHISPER_MODEL
        openai.api_key = self.api_key
        logger.info(f"✅ Whisper ASR initialized with model: {self.model}")

    def transcribe_audio_file(self, audio_file_path: str) -> Dict[str, str]:
        """
        Transcribe audio from a file using Whisper

        Args:
            audio_file_path (str): Path to audio file (mp3, wav, m4a, etc.)

        Returns:
            Dict with keys:
                - text: Transcribed text
                - language: Detected language code
        """
        try:
            logger.info(f"📝 Transcribing audio file: {audio_file_path}")

            with open(audio_file_path, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    model=self.model, file=audio_file, language=None  # Auto-detect
                )

            text = response.get("text", "")
            logger.info(f"✅ Transcription complete: {text[:100]}...")

            # Try to detect language from context
            detected_language = self._detect_language(text)

            return {
                "text": text,
                "language": detected_language,
                "confidence": 0.95,  # Whisper doesn't provide confidence
                "model": self.model,
            }

        except FileNotFoundError:
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            return {"text": "", "language": "unknown", "error": "File not found"}

        except Exception as e:
            logger.error(f"❌ Transcription error: {str(e)}")
            return {"text": "", "language": "unknown", "error": str(e)}

    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str = "wav") -> Dict[str, str]:
        """
        Transcribe audio from bytes (live audio stream)

        Args:
            audio_bytes (bytes): Raw audio data
            audio_format (str): Audio format (wav, mp3, m4a)

        Returns:
            Dict with transcription results
        """
        try:
            logger.info(f"🎤 Transcribing {len(audio_bytes)} bytes of audio")

            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"

            response = openai.Audio.transcribe(
                model=self.model, file=audio_file, language=None
            )

            text = response.get("text", "")
            detected_language = self._detect_language(text)

            logger.info(f"✅ Live audio transcribed: {text[:100]}...")

            return {
                "text": text,
                "language": detected_language,
                "confidence": 0.95,
                "model": self.model,
                "audio_length_bytes": len(audio_bytes),
            }

        except Exception as e:
            logger.error(f"❌ Live transcription error: {str(e)}")
            return {"text": "", "language": "unknown", "error": str(e)}

    def _detect_language(self, text: str) -> str:
        """
        Detect language from transcribed text using simple heuristics
        In production, use a dedicated language detection library

        Args:
            text (str): Transcribed text

        Returns:
            str: ISO 639-1 language code (e.g., 'en', 'hi', 'es')
        """
        # Simple pattern matching for common languages
        language_patterns = {
            "en": r"[a-zA-Z]",  # English uses Latin script
            "hi": r"[\u0900-\u097F]",  # Devanagari script (Hindi)
            "es": r"[a-záéíóúñü]",  # Spanish accents
            "fr": r"[a-zàâäæçéèêëîïôöœù]",  # French accents
            "de": r"[a-zäöüß]",  # German umlauts
            "zh": r"[\u4E00-\u9FFF]",  # Chinese characters
        }

        for lang, pattern in language_patterns.items():
            import re

            if re.search(pattern, text, re.IGNORECASE):
                return lang

        return "unknown"

    def get_supported_languages(self) -> list:
        """
        Get list of supported languages

        Returns:
            list: ISO 639-1 language codes
        """
        return config.SUPPORTED_LANGUAGES


# Example usage
if __name__ == "__main__":
    # Initialize ASR
    asr = WhisperASR()

    # Example 1: Transcribe from file
    # result = asr.transcribe_audio_file("path/to/audio.wav")
    # print(f"Transcribed: {result['text']}")
    # print(f"Language: {result['language']}")

    # Example 2: Transcribe from bytes (live audio)
    # with open("path/to/audio.wav", "rb") as f:
    #     audio_data = f.read()
    # result = asr.transcribe_audio_bytes(audio_data, "wav")
    # print(result)

    print("✅ ASR module loaded successfully!")