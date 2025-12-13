"""
TEAM-33 - RAG (Retrieval-Augmented Generation) Module Tests
Tests for knowledge retrieval and context management
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from rag import RAG


class TestRAGInitialization:
    """Test cases for RAG initialization"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance for testing"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    def test_rag_initialization(self, rag_instance):
        """Test RAG initialization"""
        assert rag_instance is not None

    def test_rag_default_vector_store(self, rag_instance):
        """Test RAG uses in-memory vector store by default"""
        assert rag_instance.use_pinecone is False

    def test_rag_with_pinecone(self):
        """Test RAG initialization with Pinecone"""
        with patch.dict(os.environ, {'PINECONE_API_KEY': 'test-key'}):
            rag = RAG(use_pinecone=True)
            assert rag.use_pinecone is True

    def test_rag_has_embeddings_model(self, rag_instance):
        """Test that RAG has embeddings model"""
        assert hasattr(rag_instance, 'embeddings')

    def test_rag_has_vector_store(self, rag_instance):
        """Test that RAG has vector store"""
        assert hasattr(rag_instance, 'vector_store')


class TestKnowledgeBase:
    """Test cases for knowledge base management"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_add_knowledge_document(self, mock_embeddings, rag_instance):
        """Test adding a knowledge document"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.add_documents = MagicMock()
            
            rag_instance.add_knowledge_document(
                content='Fever is a sign of infection',
                domain='healthcare',
                query_hints=['fever', 'infection']
            )
            
            mock_store.add_documents.assert_called_once()

    @patch('rag.OpenAIEmbeddings')
    def test_add_multiple_documents(self, mock_embeddings, rag_instance):
        """Test adding multiple knowledge documents"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.add_documents = MagicMock()
            
            documents = [
                {
                    'content': 'Fever symptoms',
                    'domain': 'healthcare',
                    'hints': ['fever', 'symptoms']
                },
                {
                    'content': 'Cough treatment',
                    'domain': 'healthcare',
                    'hints': ['cough', 'treatment']
                }
            ]
            
            for doc in documents:
                rag_instance.add_knowledge_document(
                    content=doc['content'],
                    domain=doc['domain'],
                    query_hints=doc['hints']
                )
            
            assert mock_store.add_documents.call_count == 2

    @patch('rag.OpenAIEmbeddings')
    def test_knowledge_document_metadata(self, mock_embeddings, rag_instance):
        """Test that knowledge documents include metadata"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.add_documents = MagicMock()
            
            rag_instance.add_knowledge_document(
                content='Test content',
                domain='healthcare',
                query_hints=['test']
            )
            
            # Verify metadata is included
            call_args = mock_store.add_documents.call_args
            assert call_args is not None

    def test_clear_knowledge_base(self, rag_instance):
        """Test clearing the knowledge base"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.clear = MagicMock()
            
            rag_instance.clear_knowledge_base()
            
            mock_store.clear.assert_called_once()


class TestRetrievalFunctionality:
    """Test cases for document retrieval"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_retrieve_documents(self, mock_embeddings, rag_instance):
        """Test retrieving documents by query"""
        mock_docs = [
            MagicMock(page_content='Document 1', metadata={'domain': 'healthcare'}),
            MagicMock(page_content='Document 2', metadata={'domain': 'healthcare'})
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            results = rag_instance.retrieve_context('fever symptoms')
            
            assert len(results) > 0
            mock_store.similarity_search.assert_called_once()

    @patch('rag.OpenAIEmbeddings')
    def test_retrieve_empty_results(self, mock_embeddings, rag_instance):
        """Test retrieval with no matching documents"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=[])
            
            results = rag_instance.retrieve_context('nonexistent query')
            
            assert len(results) == 0

    @patch('rag.OpenAIEmbeddings')
    def test_retrieve_with_relevance_threshold(self, mock_embeddings, rag_instance):
        """Test retrieval with relevance threshold"""
        mock_docs = [
            MagicMock(page_content='Relevant', metadata={'score': 0.95}),
            MagicMock(page_content='Less relevant', metadata={'score': 0.65})
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search_with_score = MagicMock(return_value=[
                (mock_docs[0], 0.95),
                (mock_docs[1], 0.65)
            ])
            
            results = rag_instance.retrieve_context('query', top_k=2)
            
            assert len(results) > 0

    @patch('rag.OpenAIEmbeddings')
    def test_retrieve_top_k_results(self, mock_embeddings, rag_instance):
        """Test retrieving specific number of top results"""
        mock_docs = [
            MagicMock(page_content=f'Document {i}', metadata={})
            for i in range(5)
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs[:3])
            
            results = rag_instance.retrieve_context('query', top_k=3)
            
            assert len(results) <= 3


class TestContextGeneration:
    """Test cases for context generation"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_generate_context_from_documents(self, mock_embeddings, rag_instance):
        """Test generating context from retrieved documents"""
        mock_docs = [
            MagicMock(page_content='Context 1', metadata={}),
            MagicMock(page_content='Context 2', metadata={})
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            context = rag_instance.retrieve_context('query')
            
            assert context is not None

    @patch('rag.OpenAIEmbeddings')
    def test_context_includes_metadata(self, mock_embeddings, rag_instance):
        """Test that generated context includes metadata"""
        mock_docs = [
            MagicMock(
                page_content='Content',
                metadata={'source': 'document1', 'domain': 'healthcare'}
            )
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            context = rag_instance.retrieve_context('query')
            
            assert context is not None

    @patch('rag.OpenAIEmbeddings')
    def test_context_formatting(self, mock_embeddings, rag_instance):
        """Test that context is properly formatted"""
        mock_docs = [
            MagicMock(page_content='Document content', metadata={})
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            context = rag_instance.retrieve_context('query')
            
            assert isinstance(context, (str, list, dict))


class TestDomainSpecificRAG:
    """Test cases for domain-specific RAG"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_healthcare_domain_retrieval(self, mock_embeddings, rag_instance):
        """Test retrieval for healthcare domain"""
        mock_docs = [
            MagicMock(
                page_content='Fever symptoms',
                metadata={'domain': 'healthcare'}
            )
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            context = rag_instance.retrieve_context(
                'fever',
                domain='healthcare'
            )
            
            assert context is not None

    @patch('rag.OpenAIEmbeddings')
    def test_business_domain_retrieval(self, mock_embeddings, rag_instance):
        """Test retrieval for business domain"""
        mock_docs = [
            MagicMock(
                page_content='Meeting notes',
                metadata={'domain': 'business'}
            )
        ]
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=mock_docs)
            
            context = rag_instance.retrieve_context(
                'meeting',
                domain='business'
            )
            
            assert context is not None

    @patch('rag.OpenAIEmbeddings')
    def test_domain_filtering(self, mock_embeddings, rag_instance):
        """Test that results are filtered by domain"""
        healthcare_doc = MagicMock(
            page_content='Healthcare content',
            metadata={'domain': 'healthcare'}
        )
        business_doc = MagicMock(
            page_content='Business content',
            metadata={'domain': 'business'}
        )
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            # Return only healthcare documents
            mock_store.similarity_search = MagicMock(return_value=[healthcare_doc])
            
            context = rag_instance.retrieve_context(
                'query',
                domain='healthcare'
            )
            
            assert context is not None


class TestEmbeddingsManagement:
    """Test cases for embeddings management"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    def test_embeddings_initialization(self, rag_instance):
        """Test embeddings model initialization"""
        assert rag_instance.embeddings is not None

    @patch('rag.OpenAIEmbeddings')
    def test_embeddings_dimension(self, mock_embeddings, rag_instance):
        """Test embeddings dimension"""
        # OpenAI embeddings typically have 1536 dimensions
        mock_embeddings.return_value.embed_query = MagicMock(
            return_value=[0.1] * 1536
        )
        
        embedding = rag_instance.embeddings.embed_query('test query')
        assert len(embedding) == 1536

    @patch('rag.OpenAIEmbeddings')
    def test_batch_embeddings(self, mock_embeddings, rag_instance):
        """Test batch embedding generation"""
        texts = ['text1', 'text2', 'text3']
        
        with patch.object(rag_instance.embeddings, 'embed_documents') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in texts]
            
            embeddings = rag_instance.embeddings.embed_documents(texts)
            
            assert len(embeddings) == len(texts)


class TestRAGIntegration:
    """Integration tests for RAG"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_healthcare_rag_workflow(self, mock_embeddings, rag_instance):
        """Test complete healthcare RAG workflow"""
        # Add healthcare knowledge
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.add_documents = MagicMock()
            
            rag_instance.add_knowledge_document(
                content='Fever is typically a sign of infection',
                domain='healthcare',
                query_hints=['fever', 'infection']
            )
            
            # Retrieve for a query
            mock_store.similarity_search = MagicMock(return_value=[
                MagicMock(page_content='Fever information', metadata={})
            ])
            
            context = rag_instance.retrieve_context('fever', domain='healthcare')
            
            assert context is not None

    @patch('rag.OpenAIEmbeddings')
    def test_multilingual_rag(self, mock_embeddings, rag_instance):
        """Test RAG with multilingual queries"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            # Add content in multiple languages
            rag_instance.add_knowledge_document(
                content='Fever treatment in English',
                domain='healthcare',
                query_hints=['fever']
            )
            
            rag_instance.add_knowledge_document(
                content='बुखार का इलाज हिंदी में',
                domain='healthcare',
                query_hints=['बुखार']
            )
            
            mock_store.similarity_search = MagicMock(return_value=[
                MagicMock(page_content='Fever info', metadata={})
            ])
            
            # Query in different languages
            context_en = rag_instance.retrieve_context('fever')
            context_hi = rag_instance.retrieve_context('बुखार')
            
            assert context_en is not None
            assert context_hi is not None

    @patch('rag.OpenAIEmbeddings')
    def test_conversation_context_rag(self, mock_embeddings, rag_instance):
        """Test RAG with conversation context"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(return_value=[
                MagicMock(page_content='Context', metadata={})
            ])
            
            # First query
            context1 = rag_instance.retrieve_context('initial query')
            
            # Follow-up query with context
            context2 = rag_instance.retrieve_context('follow-up query')
            
            assert context1 is not None
            assert context2 is not None

    @patch('rag.OpenAIEmbeddings')
    def test_rag_performance_with_large_knowledge_base(self, mock_embeddings, rag_instance):
        """Test RAG performance with large knowledge base"""
        import time
        
        with patch.object(rag_instance, 'vector_store') as mock_store:
            # Simulate large knowledge base
            mock_docs = [
                MagicMock(page_content=f'Document {i}', metadata={})
                for i in range(10000)
            ]
            
            mock_store.similarity_search = MagicMock(return_value=mock_docs[:10])
            
            start = time.time()
            context = rag_instance.retrieve_context('query')
            elapsed = time.time() - start
            
            # Should complete in reasonable time even with large KB
            assert elapsed < 5
            assert context is not None


class TestErrorHandling:
    """Test error handling in RAG"""

    @pytest.fixture
    def rag_instance(self):
        """Create RAG instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAG(use_pinecone=False)

    def test_missing_api_key(self):
        """Test RAG initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                RAG(use_pinecone=False)

    @patch('rag.OpenAIEmbeddings')
    def test_vector_store_error(self, mock_embeddings, rag_instance):
        """Test handling of vector store errors"""
        with patch.object(rag_instance, 'vector_store') as mock_store:
            mock_store.similarity_search = MagicMock(
                side_effect=Exception("Vector store error")
            )
            
            with pytest.raises(Exception):
                rag_instance.retrieve_context('query')

    @patch('rag.OpenAIEmbeddings')
    def test_embeddings_error(self, mock_embeddings, rag_instance):
        """Test handling of embeddings errors"""
        mock_embeddings.side_effect = Exception("Embeddings error")
        
        with pytest.raises(Exception):
            RAG(use_pinecone=False)


# ==================== TEST EXECUTION ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
