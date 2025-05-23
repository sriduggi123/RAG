from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import random
import uvicorn
from pathlib import Path
import shutil

from rag_service import RAGService
from llm_manager import LLMManager
from document_processor import DocumentProcessor

app = FastAPI(title="RAG LLM Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_manager = LLMManager()
document_processor = DocumentProcessor()
rag_service = RAGService(llm_manager)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    llm_used: str

class StatusResponse(BaseModel):
    status: str
    message: str
    documents_count: int

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "RAG LLM Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "llms_available": llm_manager.get_available_llms()}

@app.post("/upload", response_model=StatusResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents for RAG"""
    try:
        uploaded_files = []
        
        for file in files:
            # Validate file type
            if not document_processor.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Save file
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(str(file_path))
        
        # Process documents
        documents = document_processor.process_documents(uploaded_files)
        
        # Add to vector store
        rag_service.add_documents(documents)
        
        # Clean up uploaded files (optional)
        for file_path in uploaded_files:
            os.remove(file_path)
        
        return StatusResponse(
            status="success",
            message=f"Successfully processed {len(files)} documents",
            documents_count=len(rag_service.get_documents())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get RAG-based answer"""
    try:
        if not rag_service.has_documents():
            raise HTTPException(
                status_code=400, 
                detail="No documents available. Please upload documents first."
            )
        
        # Get answer using RAG
        result = rag_service.get_answer(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            llm_used=result["llm_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status"""
    documents_count = len(rag_service.get_documents())
    
    return StatusResponse(
        status="ready" if documents_count > 0 else "no_documents",
        message=f"System ready with {documents_count} documents" if documents_count > 0 else "No documents uploaded",
        documents_count=documents_count
    )

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        rag_service.clear_documents()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the system"""
    try:
        documents = rag_service.get_documents()
        return {
            "documents": [{"content": doc.page_content[:100] + "...", "metadata": doc.metadata} for doc in documents],
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)