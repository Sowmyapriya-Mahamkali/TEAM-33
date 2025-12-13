"""
TEAM-33 AI Agent - Language Model (LLM) Module
Uses OpenAI GPT-4 or Anthropic Claude for intelligent responses
"""

import logging
from typing import Dict, List, Optional
import openai
from anthropic import Anthropic
from config import config

# Setup logging
logger = logging.getLogger(__name__)


class LLMTranslator:
    """Language Model module for generating intelligent responses"""

    def __init__(self, use_claude: bool = False):
        """
        Initialize LLM service

        Args:
            use_claude (bool): Use Claude instead of GPT-4
        """
        self.use_claude = use_claude
        self.conversation_history: List[Dict] = []
        self.max_history = config.CONTEXT_WINDOW_SIZE

        if use_claude:
            self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = config.CLAUDE_MODEL
            logger.info(f"✅ Claude initialized with model: {self.model}")
        else:
            openai.api_key = config.OPENAI_API_KEY
            self.model = config.OPENAI_MODEL
            logger.info(f"✅ GPT-4 initialized with model: {self.model}")

    def translate_and_respond(
        self, user_text: str, source_language: str, target_language: str
    ) -> Dict[str, str]:
        """
        Translate text and generate a contextual response

        Args:
            user_text (str): Original user input
            source_language (str): Source language code (e.g., 'hi')
            target_language (str): Target language code (e.g., 'en')

        Returns:
            Dict with:
                - translated_text: Translated input
                - response: AI-generated response
                - response_translated: Response in target language
        """
        try:
            logger.info(
                f"🔄 Translating from {source_language} to {target_language}"
            )

            # Add user message to history
            self._add_to_history("user", user_text)

            # Create context-aware prompt
            system_prompt = f"""You are a helpful multilingual AI assistant.
The user is speaking in {source_language} and wants responses in {target_language}.
Maintain context from previous messages and provide accurate, helpful responses.
Keep responses concise and natural."""

            # Generate response using LLM
            if self.use_claude:
                response = self._claude_response(user_text, system_prompt)
            else:
                response = self._gpt_response(user_text, system_prompt)

            # Add response to history
            self._add_to_history("assistant", response)

            logger.info(f"✅ Response generated: {response[:100]}...")

            return {
                "original_text": user_text,
                "source_language": source_language,
                "target_language": target_language,
                "response": response,
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"❌ Translation error: {str(e)}")
            return {
                "original_text": user_text,
                "response": "",
                "error": str(e),
            }

    def _gpt_response(self, user_text: str, system_prompt: str) -> str:
        """Generate response using OpenAI GPT-4"""
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in self.conversation_history[-self.max_history :]:
            messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.7,  # Creativity level
            max_tokens=200,
            top_p=0.9,
        )

        return response["choices"][0]["message"]["content"]

    def _claude_response(self, user_text: str, system_prompt: str) -> str:
        """Generate response using Anthropic Claude"""
        messages = [{"role": "user", "content": user_text}]

        # Add conversation history
        for msg in self.conversation_history[-self.max_history :]:
            messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

        # Keep only recent messages to manage memory
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("📋 Conversation history cleared")

    def get_context(self) -> List[Dict]:
        """Get current conversation context"""
        return self.conversation_history.copy()

    def healthcare_response(self, user_text: str) -> Dict[str, str]:
        """
        Specialized healthcare assistant response

        Args:
            user_text (str): Patient/user medical question

        Returns:
            Dict with medical assistant response
        """
        system_prompt = """You are a helpful medical information assistant.
Provide accurate, empathetic health information.
IMPORTANT: Always remind users to consult with a licensed healthcare provider
for medical diagnosis and treatment decisions.
Be clear this is informational only, not medical advice."""

        try:
            self._add_to_history("user", user_text)

            if self.use_claude:
                response = self._claude_response(user_text, system_prompt)
            else:
                response = self._gpt_response(user_text, system_prompt)

            self._add_to_history("assistant", response)

            return {
                "medical_query": user_text,
                "assistant_response": response,
                "disclaimer": "For medical emergencies, contact emergency services immediately.",
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"❌ Healthcare response error: {str(e)}")
            return {"error": str(e), "assistant_response": ""}


# Example usage
if __name__ == "__main__":
    # Initialize LLM with GPT-4
    llm = LLMTranslator(use_claude=False)

    # Example translation and response
    # result = llm.translate_and_respond(
    #     "मुझे डॉक्टर की जरूरत है",
    #     source_language="hi",
    #     target_language="en"
    # )
    # print(result)

    # Example healthcare response
    # health_result = llm.healthcare_response("मुझे गले में दर्द है")
    # print(health_result)

    print("✅ LLM module loaded successfully!")