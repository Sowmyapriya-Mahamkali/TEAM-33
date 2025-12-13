"""
TEAM-33 AI Agent - RAG (Retrieval-Augmented Generation) Module
Provides context-aware knowledge for accurate multilingual responses
"""

import logging
from typing import Dict, List, Optional
import numpy as np

# Uncomment to use actual vector databases
# from pinecone import Pinecone
# import weaviate

from config import config

# Setup logging
logger = logging.getLogger(__name__)


class RAG:
    """Retrieval-Augmented Generation for context-aware responses"""

    def __init__(self, use_pinecone: bool = False):
        """
        Initialize RAG system

        Args:
            use_pinecone (bool): Use Pinecone for vector storage
        """
        self.use_pinecone = use_pinecone
        self.knowledge_base: List[Dict] = []
        self.embeddings_cache: Dict = {}

        if use_pinecone:
            self._init_pinecone()
        else:
            self._init_local_knowledge_base()

        logger.info("✅ RAG module initialized")

    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            from pinecone import Pinecone

            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = self.pc.Index(config.PINECONE_INDEX)
            logger.info(f"✅ Pinecone connected to index: {config.PINECONE_INDEX}")
        except ImportError:
            logger.warning("⚠️ Pinecone not installed. Using local knowledge base.")
            self.use_pinecone = False
            self._init_local_knowledge_base()

    def _init_local_knowledge_base(self):
        """Initialize local knowledge base (fallback)"""
        # Sample healthcare knowledge base for demonstration
        self.knowledge_base = [
            {
                "id": 1,
                "domain": "healthcare",
                "query": "fever symptoms",
                "content": "High temperature, body aches, fatigue. Drink water, rest, consult doctor if >103°F",
            },
            {
                "id": 2,
                "domain": "healthcare",
                "query": "sore throat",
                "content": "Throat pain, difficulty swallowing. Gargle salt water, drink warm tea, rest voice",
            },
            {
                "id": 3,
                "domain": "healthcare",
                "query": "headache relief",
                "content": "Pain in head/temples. Rest in quiet dark room, drink water, apply cold compress",
            },
            {
                "id": 4,
                "domain": "emergency",
                "query": "chest pain",
                "content": "EMERGENCY: Call ambulance immediately. Chest pain can indicate heart problem.",
            },
            {
                "id": 5,
                "domain": "nutrition",
                "query": "healthy diet",
                "content": "Balanced diet: fruits, vegetables, whole grains, lean proteins, healthy fats",
            },
        ]
        logger.info(f"✅ Local knowledge base initialized with {len(self.knowledge_base)} entries")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context for a query

        Args:
            query (str): User query or transcribed text
            top_k (int): Number of results to return

        Returns:
            List of relevant documents with scores
        """
        try:
            logger.info(f"🔍 Retrieving context for: {query[:50]}...")

            if self.use_pinecone:
                return self._retrieve_from_pinecone(query, top_k)
            else:
                return self._retrieve_from_local(query, top_k)

        except Exception as e:
            logger.error(f"❌ Retrieval error: {str(e)}")
            return []

    def _retrieve_from_local(self, query: str, top_k: int) -> List[Dict]:
        """Simple similarity search in local knowledge base"""
        query_lower = query.lower()

        # Simple keyword matching (in production, use semantic similarity)
        results = []
        for doc in self.knowledge_base:
            score = 0
            doc_text = f"{doc['query']} {doc['content']}".lower()

            # Calculate simple relevance score
            for word in query_lower.split():
                if word in doc_text:
                    score += 1

            if score > 0:
                results.append({"id": doc["id"], "content": doc["content"], "score": score})

        # Sort by score and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"✅ Retrieved {len(results[:top_k])} documents")
        return results[:top_k]

    def _retrieve_from_pinecone(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve from Pinecone vector database"""
        try:
            # Generate embeddings for query (would use OpenAI embeddings)
            query_embedding = self._get_embedding(query)

            # Search Pinecone
            results = self.index.query(
                vector=query_embedding, top_k=top_k, include_metadata=True
            )

            # Format results
            formatted_results = [
                {"id": match["id"], "content": match["metadata"]["text"], "score": match["score"]}
                for match in results["matches"]
            ]

            logger.info(f"✅ Retrieved {len(formatted_results)} documents from Pinecone")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Pinecone retrieval error: {str(e)}")
            return []

    def add_to_knowledge_base(self, document: Dict) -> bool:
        """
        Add new document to knowledge base

        Args:
            document (Dict): Document with keys: id, domain, content

        Returns:
            bool: Success status
        """
        try:
            if self.use_pinecone:
                return self._add_to_pinecone(document)
            else:
                self.knowledge_base.append(document)
                logger.info(f"✅ Document added to local knowledge base: {document.get('id')}")
                return True

        except Exception as e:
            logger.error(f"❌ Error adding document: {str(e)}")
            return False

    def _add_to_pinecone(self, document: Dict) -> bool:
        """Add document to Pinecone"""
        try:
            embedding = self._get_embedding(document["content"])
            self.index.upsert(
                vectors=[
                    (
                        str(document["id"]),
                        embedding,
                        {"text": document["content"], "domain": document.get("domain", "general")},
                    )
                ]
            )
            logger.info(f"✅ Document added to Pinecone: {document.get('id')}")
            return True
        except Exception as e:
            logger.error(f"❌ Pinecone add error: {str(e)}")
            return False

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text (using OpenAI)
        In production, use proper embedding models

        Args:
            text (str): Text to embed

        Returns:
            List[float]: Embedding vector
        """
        # For demo purposes, return dummy embedding
        # In production: use openai.Embedding.create() or similar
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1536).tolist()

    def augment_prompt(self, query: str, llm_input: str) -> str:
        """
        Augment LLM input with retrieved context

        Args:
            query (str): Original user query
            llm_input (str): Input for LLM

        Returns:
            str: Enhanced prompt with context
        """
        # Retrieve relevant documents
        context_docs = self.retrieve_context(query, top_k=3)

        if not context_docs:
            return llm_input

        # Build context string
        context_str = "Relevant Context:\n"
        for doc in context_docs:
            context_str += f"- {doc['content']}\n"

        # Augment original prompt
        augmented_prompt = f"""{context_str}

Based on the above context, answer: {llm_input}"""

        logger.info(f"✅ Prompt augmented with {len(context_docs)} context documents")
        return augmented_prompt

    def get_healthcare_context(self, symptom: str) -> Dict:
        """
        Get healthcare-specific context

        Args:
            symptom (str): Patient symptom

        Returns:
            Dict with medical context and recommendations
        """
        context_docs = self.retrieve_context(symptom, top_k=2)

        if context_docs:
            return {
                "symptom": symptom,
                "medical_context": context_docs[0].get("content"),
                "recommendations": self._get_healthcare_recommendations(symptom),
                "emergency": self._is_emergency(symptom),
            }
        else:
            return {
                "symptom": symptom,
                "medical_context": "Consult with healthcare professional",
                "emergency": self._is_emergency(symptom),
            }

    def _get_healthcare_recommendations(self, symptom: str) -> List[str]:
        """Get healthcare recommendations"""
        return [
            "Consult with a licensed healthcare provider",
            "Monitor symptoms for any changes",
            "Stay hydrated and get adequate rest",
        ]

    def _is_emergency(self, symptom: str) -> bool:
        """Check if symptom is emergency"""
        emergency_keywords = ["chest pain", "difficulty breathing", "unconscious", "severe bleeding"]
        symptom_lower = symptom.lower()
        return any(keyword in symptom_lower for keyword in emergency_keywords)


# Example usage
if __name__ == "__main__":
    # Initialize RAG
    rag = RAG(use_pinecone=False)

    # Example retrieval
    # context = rag.retrieve_context("fever symptoms")
    # print(context)

    # Example healthcare context
    # health_context = rag.get_healthcare_context("sore throat")
    # print(health_context)

    print("✅ RAG module loaded successfully!")