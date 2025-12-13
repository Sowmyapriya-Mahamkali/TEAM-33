from flask import Flask, request, jsonify
from datetime import datetime
import os
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversation_history = []
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "name": "TEAM-33 AI Agent API",
        "version": "1.0.0",
        "description": "GenAI-Powered Real-Time Multilingual AI Agent",
        "environment": ENVIRONMENT,
        "docs": "/docs",
        "health": "/health",
        "main_endpoints": {
            "health": "/health",
            "transcribe": "/api/v1/transcribe",
            "translate": "/api/v1/translate",
            "process": "/api/v1/process",
            "healthcare": "/api/v1/healthcare",
            "conversation": "/api/v1/conversation",
            "config": "/api/v1/config",
        },
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": ENVIRONMENT,
    })

@app.route('/api/v1/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        logger.info(f"Transcribing audio: {file.filename}")
        
        return jsonify({
            "success": True,
            "transcription": "This is a mock transcription of the audio file. In production, this would use OpenAI Whisper or Azure Speech Services.",
            "language": "en",
        })
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/translate', methods=['POST'])
def translate_text():
    try:
        text = request.form.get('text')
        source_language = request.form.get('source_language', 'auto')
        target_language = request.form.get('target_language', 'en')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        logger.info(f"Translating from {source_language} to {target_language}")
        
        return jsonify({
            "success": True,
            "original_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "translated_text": f"[Mock translation of: {text}]",
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/process', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        target_language = request.form.get('target_language', 'en')
        with_tts = request.form.get('with_tts', False)
        
        if not file.filename:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        logger.info(f"Processing audio pipeline: {file.filename}")
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "transcription": "Mock transcription from audio",
            "translation": f"Mock translation to {target_language}",
            "tts_generated": with_tts,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/healthcare', methods=['POST'])
def healthcare_consultation():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        patient_name = request.form.get('patient_name', 'Patient')
        
        if not file.filename:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        logger.info(f"Healthcare consultation for {patient_name}")
        
        conversation_history.append({
            "role": "user",
            "content": f"Healthcare query from {patient_name}",
            "timestamp": datetime.utcnow().isoformat(),
        })
        conversation_history.append({
            "role": "assistant",
            "content": "Mock healthcare assessment. In production, this would use Claude or GPT-4.",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return jsonify({
            "success": True,
            "patient": patient_name,
            "assessment": "Mock healthcare assessment",
            "recommendations": [
                "Consult with a healthcare professional",
                "Monitor symptoms",
                "Follow-up in 7 days"
            ],
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.error(f"Healthcare error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/conversation', methods=['GET'])
def get_conversation_history():
    return jsonify({
        "success": True,
        "history": conversation_history,
        "total_messages": len(conversation_history),
    })

@app.route('/api/v1/conversation', methods=['POST'])
def add_conversation_message():
    try:
        role = request.form.get('role')
        content = request.form.get('content')
        
        if role not in ["user", "assistant"]:
            return jsonify({"success": False, "error": "Invalid role"}), 400
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        conversation_history.append(message)
        
        return jsonify({
            "success": True,
            "message": message,
            "total_messages": len(conversation_history),
        })
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/conversation', methods=['DELETE'])
def clear_conversation():
    conversation_history.clear()
    return jsonify({
        "success": True,
        "message": "Conversation cleared",
    })

@app.route('/api/v1/config', methods=['GET'])
def get_configuration():
    return jsonify({
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "supported_languages": 99,
        "default_language": "en",
        "context_window": 4096,
        "sample_rate": 16000,
        "models": {
            "speech": "whisper-1",
            "llm": "gpt-4",
            "claude": "claude-3-opus",
            "tts_voice": "en-US-AriaNeural",
        },
    })

@app.route('/api/v1/languages', methods=['GET'])
def get_supported_languages():
    languages = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "hi", "ar"]
    return jsonify({
        "supported_languages": languages,
        "total": len(languages),
        "description": "List of ISO 639-1 language codes",
    })

@app.route('/api/v1/knowledge/add', methods=['POST'])
def add_knowledge():
    try:
        title = request.form.get('title')
        content = request.form.get('content')
        
        logger.info(f"Adding document: {title}")
        return jsonify({
            "success": True,
            "document": title,
            "message": "Document added to knowledge base (mock)",
        })
    except Exception as e:
        logger.error(f"Knowledge add error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/knowledge/search', methods=['GET'])
def search_knowledge():
    try:
        query = request.args.get('query')
        logger.info(f"Searching knowledge base: {query}")
        return jsonify({
            "success": True,
            "query": query,
            "results": [
                {
                    "title": "Mock Document 1",
                    "relevance": 0.95,
                    "snippet": "This is a mock search result..."
                }
            ],
            "total_results": 1,
        })
    except Exception as e:
        logger.error(f"Knowledge search error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/demo', methods=['GET'])
def demo_endpoint():
    return jsonify({
        "message": "Welcome to TEAM-33 AI Agent API",
        "examples": {
            "1_transcribe": {
                "method": "POST",
                "endpoint": "/api/v1/transcribe",
                "curl": 'curl -X POST "http://localhost:8000/api/v1/transcribe" -F "file=@audio.wav"',
            },
            "2_translate": {
                "method": "POST",
                "endpoint": "/api/v1/translate",
                "curl": 'curl -X POST "http://localhost:8000/api/v1/translate" -F "text=Hello" -F "target_language=es"',
            },
            "3_full_pipeline": {
                "method": "POST",
                "endpoint": "/api/v1/process",
                "curl": 'curl -X POST "http://localhost:8000/api/v1/process" -F "file=@audio.wav" -F "target_language=en"',
            },
            "4_healthcare": {
                "method": "POST",
                "endpoint": "/api/v1/healthcare",
                "curl": 'curl -X POST "http://localhost:8000/api/v1/healthcare" -F "file=@patient_audio.wav"',
            },
            "5_conversation": {
                "method": "GET",
                "endpoint": "/api/v1/conversation",
                "curl": 'curl "http://localhost:8000/api/v1/conversation"',
            },
        },
        "docs": "Visit http://localhost:8000/api/v1/demo for full API documentation",
    })

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled error: {str(e)}", exc_info=True)
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "detail": str(e) if DEBUG else "Server error",
    }), 500

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("TEAM-33 AI Agent Backend Server Starting")
    logger.info("=" * 50)
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Debug: {DEBUG}")
    logger.info("=" * 50)
    logger.info("API: http://localhost:8000")
    logger.info("Demo: http://localhost:8000/api/v1/demo")
    logger.info("=" * 50)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
