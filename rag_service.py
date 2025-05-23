from typing import List, Dict, Any
import logging
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
import os

from llm_manager import LLMManager

logger = logging.getLogger(__name__)

class RAGService:
    """Handles RAG (Retrieval-Augmented Generation) operations"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.vector_store = None
        self.documents: List[Document] = []
        
        # Initialize embeddings (using OpenAI by default, fallback to alternatives)
        self._initialize_embeddings()
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Custom prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            # Try OpenAI embeddings first
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.embeddings = OpenAIEmbeddings(
                    api_key=openai_key,
                    model="text-embedding-ada-002"
                )
                logger.info("OpenAI embeddings initialized")
            else:
                # Fallback to other embedding options if needed
                raise ValueError("No embedding API key found")
                
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            # Initialize empty Chroma vector store
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _create_prompt_template(self):
        """Create custom prompt template for RAG"""
        template = """You are a helpful AI assistant that answers questions based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain relevant information to answer the question, respond with "Nothing relevant found."
3. Be concise but comprehensive in your answer.
4. If you reference specific information, indicate which part of the context it comes from.

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Store documents locally for reference
            self.documents.extend(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def get_answer(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Get RAG-based answer for a question"""
        try:
            if not self.has_documents():
                raise ValueError("No documents available in vector store")
            
            # Get random LLM
            llm, llm_name = self.llm_manager.get_random_llm()
            
            # Retrieve relevant documents
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            relevant_docs = retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "Nothing relevant found.",
                    "sources": [],
                    "llm_used": llm_name
                }
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate answer using selected LLM
            prompt = self.prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            
            # Extract sources
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
            
            # Check if the answer indicates no relevant information found
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer_text,
                "sources": sources,
                "llm_used": llm_name,
                "retrieved_docs_count": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def has_documents(self) -> bool:
        """Check if vector store has documents"""
        return len(self.documents) > 0
    
    def get_documents(self) -> List[Document]:
        """Get all documents"""
        return self.documents.copy()
    
    def clear_documents(self):
        """Clear all documents from vector store"""
        try:
            # Clear the vector store
            if self.vector_store:
                # Delete the collection and reinitialize
                self.vector_store.delete_collection()
                self._initialize_vector_store()
            
            # Clear local document storage
            self.documents.clear()
            
            logger.info("All documents cleared from vector store")
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise
    
    def get_similar_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get similar documents for a query"""
        try:
            if not self.has_documents():
                return []
            
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            return retriever.get_relevant_documents(query)
            
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "total_documents": len(self.documents),
            "available_llms": self.llm_manager.get_available_llms(),
            "vector_store_initialized": self.vector_store is not None,
            "embeddings_model": "text-embedding-ada-002"
        }