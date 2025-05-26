from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import random
import uvicorn
from pathlib import Path
import shutil
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, auth
from rag_service import RAGService
from llm_manager import LLMManager
from document_processor import DocumentProcessor

load_dotenv()
app = FastAPI(title="RAG LLM Backend", version="1.0.0")

cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
firebase_admin.initialize_app(cred)

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

# Authentication dependency
async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    token = authorization.split(' ')[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.get("/")
async def root():
    return {"message": "RAG LLM Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "llms_available": llm_manager.get_available_llms()}

@app.post("/upload", response_model=StatusResponse)
async def upload_documents(files: List[UploadFile] = File(...), user_id: str = Depends(get_current_user)):
    try:
        uploaded_files = []
        
        for file in files:
            # Validate file type
            if not document_processor.is_supported_file(file.filename):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(str(file_path))
        
        # Process documents
        documents = document_processor.process_documents(uploaded_files)
        rag_service.add_documents(user_id, documents)
        
        # Clean up uploaded files (optional)
        for file_path in uploaded_files:
            os.remove(file_path)
        
        return StatusResponse(
            status="success",
            message=f"Successfully processed {len(files)} documents",
            documents_count=rag_service.get_document_count(user_id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
    
    
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, user_id: str = Depends(get_current_user)):
    try:
        if not rag_service.has_documents(user_id):
            raise HTTPException(status_code=400, detail="No documents available. Please upload documents first.")
        result = rag_service.get_answer(user_id, request.question)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
    

@app.get("/status", response_model=StatusResponse)
async def get_status(user_id: str = Depends(get_current_user)):
    documents_count = rag_service.get_document_count(user_id)
    return StatusResponse(
        status="ready" if documents_count > 0 else "no_documents",
        message=f"System ready with {documents_count} documents" if documents_count > 0 else "No documents uploaded",
        documents_count=documents_count
    )

@app.delete("/documents")
async def clear_documents(user_id: str = Depends(get_current_user)):
    try:
        rag_service.clear_documents(user_id)
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")
    
@app.get("/documents")
async def list_documents(user_id: str = Depends(get_current_user)):
    try:
        documents = rag_service.get_user_documents(user_id)
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)