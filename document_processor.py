import os
from typing import List
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, parsing, and text splitting"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.md': self._load_markdown
        }
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        extension = Path(filename).suffix.lower()
        return extension in self.supported_extensions
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple documents and return split text chunks"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                # Load document
                documents = self._load_document(file_path)
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(documents)
                
                # Add source metadata
                for doc in split_docs:
                    doc.metadata['source'] = os.path.basename(file_path)
                    doc.metadata['file_path'] = file_path
                
                all_documents.extend(split_docs)
                logger.info(f"Processed {file_path}: {len(split_docs)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                raise
        
        return all_documents
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file type"""
        extension = Path(file_path).suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader_func = self.supported_extensions[extension]
        return loader_func(file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document"""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text document"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            try:
                loader = TextLoader(file_path, encoding='latin-1')
                return loader.load()
            except Exception as e:
                logger.error(f"Error loading text file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX document"""
        try:
            loader = Docx2txtLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load Markdown document"""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading Markdown {file_path}: {e}")
            raise
    
    def get_document_info(self, file_path: str) -> dict:
        """Get basic information about a document"""
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            extension = Path(file_path).suffix.lower()
            
            return {
                'name': file_name,
                'size': file_size,
                'extension': extension,
                'supported': self.is_supported_file(file_name)
            }
        except Exception as e:
            logger.error(f"Error getting document info for {file_path}: {e}")
            return {
                'name': os.path.basename(file_path),
                'size': 0,
                'extension': '',
                'supported': False,
                'error': str(e)
            }