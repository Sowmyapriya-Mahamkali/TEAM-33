"""
TEAM-33 AI Agent - Main Controller
Orchestrates all modules (ASR, LLM, TTS, RAG)
"""

import logging
from typing import Dict, Optional
from asr import WhisperASR
from translation import LLMTranslator
from tts import TextToSpeech
from rag import RAG
from config import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AIAgent:
    """Main AI Agent Controller - Orchestrates all components"""

    def __init__(self, use_claude: bool = False, tts_provider: str = "azure", use_rag: bool = True):
        """
        Initialize the complete AI Agent

        Args:
            use_claude (bool): Use Claude instead of GPT-4
            tts_provider (str): TTS provider ('azure' or 'google')
            use_rag (bool): Enable Retrieval-Augmented Generation
        """
        logger.info("🚀 Initializing TEAM-33 AI Agent...")

        # Initialize all modules
        self.asr = WhisperASR()
        self.llm = LLMTranslator(use_claude=use_claude)
        self.tts = TextToSpeech(provider=tts_provider)
        self.rag = RAG(use_pinecone=False) if use_rag else None

        self.use_rag = use_rag
        self.conversation_context: Dict = {}

        logger.info("✅ TEAM-33 AI Agent initialized successfully!")

    def process_audio_file(
        self, audio_file_path: str, target_language: str = "en", use_tts: bool = True
    ) -> Dict:
        """
        Complete pipeline: Audio → Text → Response → Speech

        Args:
            audio_file_path (str): Path to audio file
            target_language (str): Target language for response
            use_tts (bool): Generate audio response

        Returns:
            Dict with complete conversation result
        """
        logger.info(f"🎤 Processing audio file: {audio_file_path}")

        try:
            # Step 1: Speech-to-Text
            logger.info("📝 Step 1: Converting speech to text...")
            asr_result = self.asr.transcribe_audio_file(audio_file_path)

            if not asr_result.get("text"):
                return {"error": "Failed to transcribe audio", "details": asr_result}

            user_text = asr_result["text"]
            source_language = asr_result.get("language", "unknown")

            logger.info(f"✅ Transcribed: {user_text}")
            logger.info(f"📍 Detected language: {source_language}")

            # Step 2: Augment with RAG if enabled
            if self.use_rag:
                logger.info("🔍 Step 2: Retrieving contextual knowledge...")
                augmented_prompt = self.rag.augment_prompt(user_text, user_text)
            else:
                augmented_prompt = user_text

            # Step 3: Language Model Response
            logger.info("🧠 Step 3: Generating intelligent response...")
            llm_result = self.llm.translate_and_respond(
                augmented_prompt, source_language, target_language
            )

            if not llm_result.get("response"):
                return {"error": "Failed to generate response", "details": llm_result}

            response_text = llm_result["response"]
            logger.info(f"✅ Generated response: {response_text[:100]}...")

            # Step 4: Text-to-Speech (optional)
            result = {
                "input_audio": audio_file_path,
                "transcribed_text": user_text,
                "source_language": source_language,
                "target_language": target_language,
                "ai_response": response_text,
                "conversation_id": self._generate_session_id(),
            }

            if use_tts:
                logger.info("🔊 Step 4: Converting response to speech...")
                output_file = f"response_{result['conversation_id']}.wav"
                tts_result = self.tts.synthesize(response_text, output_file)

                if tts_result.get("success"):
                    result["output_audio"] = output_file
                    logger.info(f"✅ Audio response saved: {output_file}")
                else:
                    logger.warning(f"⚠️ TTS failed: {tts_result.get('error')}")
                    result["tts_error"] = tts_result.get("error")

            logger.info("✅ Complete pipeline executed successfully!")
            return result

        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            return {"error": str(e), "status": "failed"}

    def process_live_audio(
        self, audio_bytes: bytes, audio_format: str = "wav", target_language: str = "en"
    ) -> Dict:
        """
        Process live/streaming audio

        Args:
            audio_bytes (bytes): Raw audio data
            audio_format (str): Audio format
            target_language (str): Target language

        Returns:
            Dict with processing results
        """
        try:
            # Step 1: ASR from bytes
            asr_result = self.asr.transcribe_audio_bytes(audio_bytes, audio_format)

            if not asr_result.get("text"):
                return {"error": "Failed to transcribe", "details": asr_result}

            # Step 2-3: LLM and optional RAG
            user_text = asr_result["text"]
            source_language = asr_result.get("language", "unknown")

            llm_result = self.llm.translate_and_respond(
                user_text, source_language, target_language
            )

            return {
                "transcribed": user_text,
                "source_language": source_language,
                "response": llm_result.get("response"),
                "target_language": target_language,
            }

        except Exception as e:
            logger.error(f"❌ Live audio processing error: {str(e)}")
            return {"error": str(e)}

    def healthcare_consultation(self, audio_file_path: str) -> Dict:
        """
        Specialized healthcare consultation

        Args:
            audio_file_path (str): Patient audio

        Returns:
            Dict with medical consultation results
        """
        logger.info("🏥 Healthcare consultation mode enabled")

        try:
            # Transcribe
            asr_result = self.asr.transcribe_audio_file(audio_file_path)
            patient_statement = asr_result.get("text", "")

            if not patient_statement:
                return {"error": "Failed to transcribe patient statement"}

            # Get healthcare context
            if self.use_rag:
                health_context = self.rag.get_healthcare_context(patient_statement)
            else:
                health_context = None

            # Generate medical response
            healthcare_result = self.llm.healthcare_response(patient_statement)

            result = {
                "patient_statement": patient_statement,
                "medical_context": health_context,
                "assistant_response": healthcare_result.get("assistant_response"),
                "disclaimer": healthcare_result.get("disclaimer"),
                "is_emergency": health_context.get("emergency") if health_context else False,
            }

            # Emergency alert
            if result["is_emergency"]:
                logger.warning("🚨 EMERGENCY DETECTED - Contact emergency services immediately!")
                result["emergency_alert"] = "CONTACT EMERGENCY SERVICES IMMEDIATELY"

            return result

        except Exception as e:
            logger.error(f"❌ Healthcare consultation error: {str(e)}")
            return {"error": str(e)}

    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.llm.get_context()

    def clear_conversation(self) -> None:
        """Clear conversation context"""
        self.llm.clear_history()
        logger.info("📋 Conversation history cleared")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())[:8]


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = AIAgent(use_claude=False, tts_provider="azure", use_rag=True)

    # Example 1: Process audio file
    # result = agent.process_audio_file("sample_audio.wav", target_language="en")
    # print(result)

    # Example 2: Healthcare consultation
    # health_result = agent.healthcare_consultation("patient_audio.wav")
    # print(health_result)

    print("✅ AI Agent controller loaded successfully!")