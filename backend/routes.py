"""
TEAM-33 AI Agent - API Routes
Modular route handlers for FastAPI backend
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
from typing import Optional

from agent_controller import AIAgent

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["API Routes"])

# Initialize AI Agent (will be initialized in main.py)
agent: Optional[AIAgent] = None


def set_agent(ai_agent: AIAgent):
    """Set the AI agent instance"""
    global agent
    agent = ai_agent


# ==================== TRANSCRIPTION ROUTES ====================

@router.post("/transcribe", tags=["Audio Processing"])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Transcribe audio file to text

    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        language: Optional language hint

    Returns:
        Transcribed text and metadata
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        logger.info(f"📝 Transcribing audio: {file.filename}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Transcribe
        result = agent.asr.transcribe_audio_file(temp_path)

        # Cleanup
        os.unlink(temp_path)

        if not result.get("text"):
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        logger.info(f"✅ Transcription successful: {result['text'][:50]}...")

        return {
            "success": True,
            "transcribed_text": result.get("text"),
            "language": result.get("language"),
            "model": result.get("model"),
            "confidence": result.get("confidence"),
            "filename": file.filename,
        }

    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== FULL PIPELINE ROUTES ====================

@router.post("/process", tags=["Full Pipeline"])
async def process_audio(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
    with_tts: bool = Form(True),
):
    """
    Complete pipeline: Audio → Text → Response → Speech

    Args:
        file: Input audio file
        target_language: Target language for response (en, hi, es, etc.)
        with_tts: Generate audio response

    Returns:
        Transcription, response, and optional audio output
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        logger.info(f"🎤 Processing audio: {file.filename}")

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Process through full pipeline
        result = agent.process_audio_file(
            temp_path,
            target_language=target_language,
            use_tts=with_tts
        )

        # Cleanup
        os.unlink(temp_path)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        logger.info(f"✅ Processing complete: {result.get('conversation_id')}")

        return {
            "success": True,
            "transcribed_text": result.get("transcribed_text"),
            "source_language": result.get("source_language"),
            "ai_response": result.get("ai_response"),
            "target_language": result.get("target_language"),
            "session_id": result.get("conversation_id"),
            "has_audio": "output_audio" in result,
            "output_file": result.get("output_audio"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== HEALTHCARE ROUTES ====================

@router.post("/healthcare", tags=["Healthcare"])
async def healthcare_consultation(
    file: UploadFile = File(...)
):
    """
    Specialized healthcare consultation

    Args:
        file: Patient audio recording

    Returns:
        Medical consultation response with safety disclaimers
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        logger.info(f"🏥 Healthcare consultation: {file.filename}")

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Healthcare consultation
        result = agent.healthcare_consultation(temp_path)

        # Cleanup
        os.unlink(temp_path)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        logger.info(f"✅ Healthcare consultation complete")

        return {
            "success": True,
            "patient_statement": result.get("patient_statement"),
            "assistant_response": result.get("assistant_response"),
            "is_emergency": result.get("is_emergency", False),
            "emergency_alert": result.get("emergency_alert"),
            "medical_context": result.get("medical_context"),
            "disclaimer": "⚠️ This is informational only, not medical advice. Always consult a licensed healthcare provider for proper diagnosis and treatment.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Healthcare error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== TEXT-TO-SPEECH ROUTES ====================

@router.post("/text-to-speech", tags=["TTS"])
async def text_to_speech(
    text: str = Form(...),
    provider: str = Form("azure")
):
    """
    Convert text to speech

    Args:
        text: Text to convert to speech
        provider: TTS provider ('azure' or 'google')

    Returns:
        Audio file as response
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        logger.info(f"🔊 Text-to-speech: {text[:50]}...")

        # Generate speech
        output_file = f"tts_output_{id(text)}.wav"
        result = agent.tts.synthesize(text, output_file)

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {result.get('error')}"
            )

        if os.path.exists(output_file):
            logger.info(f"✅ TTS successful")
            return FileResponse(
                output_file,
                media_type="audio/wav",
                filename="response.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="Audio file not created")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== CONVERSATION ROUTES ====================

@router.get("/conversation", tags=["Conversation"])
def get_conversation_history():
    """
    Get conversation history

    Returns:
        List of conversation messages with roles and content
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        history = agent.get_conversation_history()
        logger.info(f"📋 Retrieved conversation history: {len(history)} messages")

        return {
            "success": True,
            "conversation_history": history,
            "total_messages": len(history),
        }

    except Exception as e:
        logger.error(f"❌ Error getting history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/conversation", tags=["Conversation"])
def clear_conversation():
    """
    Clear conversation history

    Returns:
        Confirmation message
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        agent.clear_conversation()
        logger.info("📋 Conversation history cleared")

        return {
            "success": True,
            "message": "Conversation history cleared successfully",
        }

    except Exception as e:
        logger.error(f"❌ Error clearing history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== TRANSLATION ROUTES ====================

@router.post("/translate", tags=["Translation"])
async def translate_text(
    text: str = Form(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en")
):
    """
    Translate text directly (without audio)

    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code

    Returns:
        Translated text and response
    """
    if not agent:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")

    try:
        logger.info(f"🔄 Translation: {source_language} → {target_language}")

        # Get translation
        result = agent.llm.translate_and_respond(
            text,
            source_language,
            target_language
        )

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        logger.info(f"✅ Translation complete")

        return {
            "success": True,
            "original_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "response": result.get("response"),
            "model": result.get("model"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Translation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== RETRIEVAL ROUTES ====================

@router.get("/knowledge/{query}", tags=["Knowledge Retrieval"])
def retrieve_knowledge(query: str, top_k: int = 3):
    """
    Retrieve knowledge from RAG system

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        Relevant knowledge documents
    """
    if not agent or not agent.rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        logger.info(f"🔍 Knowledge retrieval: {query}")

        context = agent.rag.retrieve_context(query, top_k=top_k)

        logger.info(f"✅ Retrieved {len(context)} documents")

        return {
            "success": True,
            "query": query,
            "results": context,
            "total": len(context),
        }

    except Exception as e:
        logger.error(f"❌ Retrieval error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/knowledge/add", tags=["Knowledge Retrieval"])
async def add_knowledge(
    domain: str = Form(...),
    content: str = Form(...),
    query_hint: Optional[str] = Form(None)
):
    """
    Add new document to knowledge base

    Args:
        domain: Domain category (healthcare, education, etc.)
        content: Document content
        query_hint: Optional hint for queries

    Returns:
        Confirmation and document ID
    """
    if not agent or not agent.rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        logger.info(f"📚 Adding knowledge: {domain}")

        # Create document
        doc_id = len(agent.rag.knowledge_base) + 1
        document = {
            "id": doc_id,
            "domain": domain,
            "content": content,
            "query": query_hint or content[:50],
        }

        # Add to knowledge base
        success = agent.rag.add_to_knowledge_base(document)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add knowledge")

        logger.info(f"✅ Knowledge added: {doc_id}")

        return {
            "success": True,
            "document_id": doc_id,
            "domain": domain,
            "message": "Knowledge added successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding knowledge: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== STATUS ROUTES ====================

@router.get("/status", tags=["Status"])
def get_status():
    """
    Get current system status

    Returns:
        System status and component information
    """
    if not agent:
        return {
            "status": "error",
            "agent_initialized": False,
            "components": {},
        }

    try:
        return {
            "status": "healthy",
            "agent_initialized": True,
            "components": {
                "asr": "ready",
                "llm": "ready",
                "tts": "ready",
                "rag": "ready" if agent.rag else "disabled",
            },
            "conversation_length": len(agent.get_conversation_history()),
        }

    except Exception as e:
        logger.error(f"❌ Status error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
        }