from typing import List, Dict, Any
import logging
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llm_manager import LLMManager

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.prompt_template = self._create_prompt_template()
        logger.info("RAGService initialized with HuggingFace embeddings")

    def _create_prompt_template(self):
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
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _get_vector_store(self, user_id: str):
        collection_name = f"user_{user_id}"
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def _get_common_vector_store(self):
        return Chroma(
            collection_name="common_knowledge",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def add_documents(self, user_id: str, documents: List[Document]):
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            vector_store = self._get_vector_store(user_id)
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def get_answer(self, user_id: str, question: str, k: int = 4) -> Dict[str, Any]:
        try:
            # Get vector stores
            user_vector_store = self._get_vector_store(user_id)
            common_vector_store = self._get_common_vector_store()

            # Retrieve documents with scores
            user_docs_with_scores = user_vector_store.similarity_search_with_score(question, k=k)
            common_docs_with_scores = common_vector_store.similarity_search_with_score(question, k=k)

            # Combine and sort by score (lower distance = more similar)
            all_docs_with_scores = user_docs_with_scores + common_docs_with_scores
            if not all_docs_with_scores:
                return {"answer": "Nothing relevant found.", "sources": [], "llm_used": "none"}

            all_docs_with_scores.sort(key=lambda x: x[1])  # Ascending for cosine distance
            top_k_docs_with_scores = all_docs_with_scores[:k]

            # Extract documents and build context
            top_k_docs = [doc for doc, score in top_k_docs_with_scores]
            if not top_k_docs:
                return {"answer": "Nothing relevant found.", "sources": [], "llm_used": "none"}

            context = "\n\n".join([doc.page_content for doc in top_k_docs])

            # Collect sources
            sources = set([doc.metadata.get('source', 'Unknown') for doc in top_k_docs])

            # Generate answer
            llm, llm_name = self.llm_manager.get_random_llm()
            prompt = self.prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            answer_text = response.content if hasattr(response, 'content') else str(response)

            return {
                "answer": answer_text,
                "sources": list(sources),
                "llm_used": llm_name
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


    def getAnswer(self, user_id: str, question: str, k: int = 4) -> Dict[str, Any]:
        try:
            vector_store = self._get_vector_store(user_id)
            if vector_store._collection.count() == 0:
                return {"answer": "Nothing relevant found.", "sources": [], "llm_used": "none"}
            
            llm, llm_name = self.llm_manager.get_random_llm()
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            relevant_docs = retriever.invoke(question)
            
            if not relevant_docs:
                return {"answer": "Nothing relevant found.", "sources": [], "llm_used": llm_name}
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = self.prompt_template.format(context=context, question=question)
            response = llm.invoke(prompt)
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
            
            answer_text = response.content if hasattr(response, 'content') else str(response)
            return {
                "answer": answer_text,
                "sources": sources,
                "llm_used": llm_name
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def has_documents(self, user_id: str) -> bool:
        vector_store = self._get_vector_store(user_id)
        return vector_store._collection.count() > 0

    def get_user_documents(self, user_id: str) -> List[Dict]:
        try:
            vector_store = self._get_vector_store(user_id)
            results = vector_store._collection.get(include=['metadatas'])
            documents = {}
            for metadata in results['metadatas']:
                source = metadata.get('source', 'Unknown')
                if source not in documents:
                    documents[source] = {'source': source, 'chunks': 1, 'processed': True}
                else:
                    documents[source]['chunks'] += 1
            return list(documents.values())
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def clear_documents(self, user_id: str):
        try:
            vector_store = self._get_vector_store(user_id)
            results = vector_store._collection.get()
            ids = results['ids']
            if ids:
                vector_store._collection.delete(ids=ids)
            logger.info(f"Cleared all documents for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise

    def getDocumentCount(self, user_id: str) -> int:
        vector_store = self._get_vector_store(user_id)
        return vector_store._collection.count()
    
    def getDocumentsCount(self, userId: str) -> int:
        try:
            vectorStore = self._get_vector_store(userId)
            results = vectorStore._collection.get(include=['metadatas'])
            sources = {metadata.get('source', 'Unknown') for metadata in results['metadatas']}
            return len(sources)
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
        
    def get_document_count(self, userId: str) -> int:
        try:
            vectorStore = self._get_vector_store(userId)
            results = vectorStore._collection.get(include=['metadatas'])
            if not results['metadatas']:
                logger.info(f"No metadata found for user {userId}")
                return 0
            sources = {metadata.get('source', '') for metadata in results['metadatas'] if metadata.get('source')}
            logger.info(f"User {userId} has {len(sources)} unique documents")
            return len(sources)
        except Exception as e:
            logger.error(f"Error counting documents for user {userId}: {e}")
            return 0