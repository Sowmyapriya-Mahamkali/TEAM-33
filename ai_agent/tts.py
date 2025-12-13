"""
TEAM-33 AI Agent - Text-to-Speech (TTS) Module
Uses Azure Speech Services or Google Cloud for natural speech synthesis
"""

import logging
from typing import Dict, Optional
import azure.cognitiveservices.speech as speechsdk
from google.cloud import texttospeech
from config import config

# Setup logging
logger = logging.getLogger(__name__)


class AzureTTS:
    """Text-to-Speech using Azure Speech Services"""

    def __init__(self):
        """Initialize Azure TTS service"""
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION
        )
        self.voice_name = config.AZURE_TTS_VOICE
        logger.info(f"✅ Azure TTS initialized with voice: {self.voice_name}")

    def synthesize_text(self, text: str, output_file: str = None) -> Dict:
        """
        Convert text to speech using Azure

        Args:
            text (str): Text to convert to speech
            output_file (str): Optional file path to save audio

        Returns:
            Dict with audio data and metadata
        """
        try:
            logger.info(f"🎙️ Synthesizing text: {text[:50]}...")

            # Set voice
            self.speech_config.speech_synthesis_voice_name = self.voice_name

            # Configure output
            if output_file:
                audio_config = speechsdk.audio.AudioOutputConfig(
                    filename=output_file
                )
            else:
                audio_config = speechsdk.audio.AudioOutputConfig(
                    use_default_speaker=True
                )

            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, audio_config=audio_config
            )

            # Synthesize
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"✅ Speech synthesis completed successfully")
                return {
                    "success": True,
                    "text": text,
                    "voice": self.voice_name,
                    "output_file": output_file,
                    "model": "Azure Speech Services",
                }
            else:
                logger.error(f"❌ Synthesis error: {result.error_details}")
                return {
                    "success": False,
                    "error": result.error_details,
                }

        except Exception as e:
            logger.error(f"❌ Azure TTS error: {str(e)}")
            return {"success": False, "error": str(e)}

    def set_voice(self, voice_name: str) -> None:
        """
        Change voice

        Args:
            voice_name (str): Azure voice name (e.g., 'hi-IN-SwaraNeural')
        """
        self.voice_name = voice_name
        logger.info(f"🎤 Voice changed to: {voice_name}")

    def get_available_voices(self) -> Dict:
        """
        Get available voices for different languages

        Returns:
            Dict with voice options
        """
        return {
            "hindi": ["hi-IN-SwaraNeural", "hi-IN-KaranNeural"],
            "english": [
                "en-US-AriaNeural",
                "en-US-GuyNeural",
                "en-IN-NeerjaNeural",
                "en-IN-PrabhatNeural",
            ],
            "spanish": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
            "tamil": ["ta-IN-JarvarasNeural", "ta-IN-VigneshNeural"],
            "telugu": ["te-IN-MohanNeural", "te-IN-ShrutiNeural"],
        }


class GoogleCloudTTS:
    """Text-to-Speech using Google Cloud"""

    def __init__(self):
        """Initialize Google Cloud TTS service"""
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = config.GOOGLE_TTS_LANGUAGE
        self.voice_name = f"projects/resourcenames/locations/global/voices/{config.GOOGLE_TTS_LANGUAGE}-Standard-A"
        logger.info(f"✅ Google Cloud TTS initialized")

    def synthesize_text(self, text: str, output_file: str = None) -> Dict:
        """
        Convert text to speech using Google Cloud

        Args:
            text (str): Text to convert to speech
            output_file (str): Optional file path to save audio

        Returns:
            Dict with audio data and metadata
        """
        try:
            logger.info(f"🎙️ Synthesizing text with Google Cloud: {text[:50]}...")

            # Set input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Set voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            )

            # Set audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )

            # Generate speech
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Save to file if specified
            if output_file:
                with open(output_file, "wb") as out:
                    out.write(response.audio_content)
                logger.info(f"✅ Audio saved to: {output_file}")

            return {
                "success": True,
                "text": text,
                "language": self.language_code,
                "audio_content": response.audio_content,
                "output_file": output_file,
                "model": "Google Cloud Text-to-Speech",
            }

        except Exception as e:
            logger.error(f"❌ Google Cloud TTS error: {str(e)}")
            return {"success": False, "error": str(e)}

    def set_language(self, language_code: str) -> None:
        """
        Change language

        Args:
            language_code (str): Language code (e.g., 'hi-IN', 'en-US')
        """
        self.language_code = language_code
        self.voice_name = (
            f"projects/resourcenames/locations/global/voices/{language_code}-Standard-A"
        )
        logger.info(f"🌐 Language changed to: {language_code}")


# Wrapper for easy switching between Azure and Google Cloud
class TextToSpeech:
    """Unified TTS interface"""

    def __init__(self, provider: str = "azure"):
        """
        Initialize TTS with selected provider

        Args:
            provider (str): 'azure' or 'google'
        """
        if provider.lower() == "google":
            self.provider = GoogleCloudTTS()
            self.provider_name = "Google Cloud"
        else:
            self.provider = AzureTTS()
            self.provider_name = "Azure"

        logger.info(f"✅ TTS initialized with provider: {self.provider_name}")

    def synthesize(self, text: str, output_file: str = None) -> Dict:
        """Synthesize text to speech"""
        return self.provider.synthesize_text(text, output_file)


# Example usage
if __name__ == "__main__":
    # Azure TTS example
    # azure_tts = AzureTTS()
    # result = azure_tts.synthesize_text("नमस्ते, कैसे हो?", "output.wav")
    # print(result)

    # Google Cloud TTS example
    # google_tts = GoogleCloudTTS()
    # result = google_tts.synthesize_text("Hello, how are you?", "output.mp3")
    # print(result)

    # Unified interface
    # tts = TextToSpeech(provider="azure")
    # result = tts.synthesize("Hello world", "hello.wav")
    # print(result)

    print("✅ TTS module loaded successfully!")