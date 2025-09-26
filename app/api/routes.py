"""
FastAPI routes for the financial document processing system.
"""
import os
import asyncio
import logging
from typing import List, Optional
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from app.database.connection import get_database
from app.database.models import Document, ChatSession, ResearchTask
from app.database import schemas
from app.api.dependencies import validate_file_upload, ensure_upload_directory, validate_session_id, validate_document_id
from app.services.document_processor import document_processor
from app.services.embedding_service import embedding_service
from app.services.metadata_extractor import metadata_extractor
from app.agents.deep_research_agent import deep_research_agent
from app.agents.chat_agent_with_tools import chat_agent_with_tools
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Document upload and processing endpoints
@router.post("/upload", response_model=schemas.UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_upload),
    db: Session = Depends(get_database)
):
    """
    Upload and process a financial document.
    
    This endpoint:
    1. Saves the uploaded file
    2. Creates a document record in the database
    3. Starts background processing (docling parsing, embedding, metadata extraction)
    """
    try:
        # Ensure upload directory exists
        ensure_upload_directory()
        
        # Generate unique filename
        file_id = str(uuid4())
        file_ext = os.path.splitext(file.filename)[1].lower()
        saved_filename = f"{file_id}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, saved_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        document = Document(
            filename=saved_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            mime_type=file.content_type or "application/pdf"
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        document_id = str(document.id)
        
        # Start background processing
        background_tasks.add_task(process_document_pipeline, document_id, file_path)
        
        logger.info(f"Document uploaded: {file.filename} -> {document_id}")
        
        return schemas.UploadResponse(
            document_id=document.id,
            filename=saved_filename,
            file_size=len(content),
            status="uploaded",
            processing_started=True
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


async def process_document_pipeline(document_id: str, file_path: str):
    """Background task for processing uploaded documents."""
    try:
        logger.info(f"Starting processing pipeline for document {document_id}")
        
        # Step 1: Process document with docling
        await document_processor.process_document(document_id, file_path)
        logger.info(f"Document processing completed for {document_id}")
        
        # Step 2: Generate and store embeddings
        await embedding_service.embed_document(document_id)
        logger.info(f"Embedding completed for {document_id}")
        
        # Step 3: Extract metadata
        await metadata_extractor.extract_metadata(document_id)
        logger.info(f"Metadata extraction completed for {document_id}")
        
        # Clean up temporary file if needed
        try:
            if os.path.exists(file_path) and "/tmp/" in file_path:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {str(e)}")
        
        logger.info(f"Full processing pipeline completed for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error in processing pipeline for document {document_id}: {str(e)}")
        # Update document with error status
        from app.database.connection import get_db_session
        with get_db_session() as db:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_error = f"Pipeline error: {str(e)}"


@router.get("/documents", response_model=List[schemas.DocumentSummary])
def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_database)
):
    """List all uploaded documents with basic information."""
    try:
        documents = db.query(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=schemas.Document)
def get_document(
    document_id: str = Depends(validate_document_id),
    db: Session = Depends(get_database)
):
    """Get detailed information about a specific document."""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
        )


@router.get("/documents/{document_id}/status", response_model=schemas.ProcessingStatus)
def get_document_status(
    document_id: str = Depends(validate_document_id),
    db: Session = Depends(get_database)
):
    """Get processing status of a document."""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Calculate progress percentage
        progress = 0
        if document.is_processed:
            progress += 40
        if document.is_embedded:
            progress += 40
        if document.financial_facts or document.investment_data:
            progress += 20
        
        return schemas.ProcessingStatus(
            document_id=document.id,
            filename=document.filename,
            is_processed=document.is_processed,
            is_embedded=document.is_embedded,
            processing_error=document.processing_error,
            embedding_count=document.embedding_count,
            progress_percentage=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document status: {str(e)}"
        )


# Deep Research endpoints
@router.post("/documents/{document_id}/research/start", response_model=schemas.ResearchResponse)
async def start_research(
    request: schemas.ResearchRequestBody,
    document_id: str = Depends(validate_document_id),
    db: Session = Depends(get_database)
):
    """
    Start a deep research task for a document.
    
    This creates a comprehensive content outline and analysis for the specified topic.
    """
    try:
        # Validate document exists and is processed
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if not document.is_processed or not document.is_embedded:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document must be fully processed and embedded before research"
            )
        
        # Conduct research
        start_time = asyncio.get_event_loop().time()
        result = await deep_research_agent.conduct_research(
            document_id=str(document_id),
            topic=request.topic,
            custom_query=request.custom_query
        )
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return schemas.ResearchResponse(
            task_id=result['task_id'],
            content_outline={"summary": result['summary']},
            research_findings={"summary": result['summary']},
            sources_used=result['sources'],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting research: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting research: {str(e)}"
        )


@router.get("/documents/{document_id}/research/tasks", response_model=List[schemas.ResearchTask])
def list_research_tasks(
    document_id: str = Depends(validate_document_id),
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_database)
):
    """List research tasks for a specific document."""
    try:
        # Validate document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        query = db.query(ResearchTask).filter(ResearchTask.document_id == document_id)
        
        if status_filter:
            query = query.filter(ResearchTask.status == status_filter)
        
        tasks = query.order_by(ResearchTask.created_at.desc()).offset(skip).limit(limit).all()
        return tasks
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing research tasks for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing research tasks: {str(e)}"
        )


@router.get("/documents/{document_id}/research/tasks/{task_id}", response_model=schemas.ResearchTask)
def get_research_task(
    task_id: str,
    document_id: str = Depends(validate_document_id),
    db: Session = Depends(get_database)
):
    """Get detailed information about a research task for a specific document."""
    try:
        # Validate document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get task and validate it belongs to the document
        task = db.query(ResearchTask).filter(
            ResearchTask.id == task_id,
            ResearchTask.document_id == document_id
        ).first()
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research task not found for this document"
            )
        return task
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research task {task_id} for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting research task: {str(e)}"
        )


# Chat endpoints
@router.post("/conversation", response_model=schemas.ChatResponse)
async def send_chat_message(
    request: schemas.ChatRequest,
    db: Session = Depends(get_database)
):
    """
    Send a message in a chat session with optional document context.
    
    This endpoint supports RAG (Retrieval-Augmented Generation) if document_id is provided.
    """
    try:
        # Validate document if provided
        if request.document_id:
            document = db.query(Document).filter(Document.id == request.document_id).first()
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            if not document.is_embedded:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Document must be embedded before using in chat"
                )
        
        # Process chat message
        result = await chat_agent_with_tools.chat(
            message=request.message,
            session_id=str(request.session_id) if request.session_id else None,
            document_id=str(request.document_id) if request.document_id else None
        )
        
        return schemas.ChatResponse(
            message=result['message'],
            session_id=result['session_id'],
            sources_used=result.get('sources_used'),
            tool_calls=result.get('tool_calls', []),
            response_time=result['response_time'],
            token_count=result.get('token_count')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )


@router.post("/conversation/new", response_model=schemas.ChatSession)
def create_chat_session(
    request: schemas.ChatSessionCreate,
    db: Session = Depends(get_database)
):
    """Create a new chat session."""
    try:
        # Validate document if provided
        if request.document_id:
            document = db.query(Document).filter(Document.id == request.document_id).first()
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
        
        session = ChatSession(
            document_id=request.document_id,
            session_name=request.session_name or f"Chat Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            user_id=request.user_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating chat session: {str(e)}"
        )


@router.get("/conversation/sessions", response_model=List[schemas.ChatSession])
def list_chat_sessions(
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_database)
):
    """List chat sessions with optional filtering."""
    try:
        query = db.query(ChatSession).filter(ChatSession.is_active == True)
        
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
        
        if document_id:
            query = query.filter(ChatSession.document_id == document_id)
        
        sessions = query.order_by(ChatSession.last_activity.desc()).offset(skip).limit(limit).all()
        return sessions
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing chat sessions: {str(e)}"
        )


@router.get("/conversation/{session_id}/history", response_model=List[schemas.ChatMessage])
async def get_chat_history(
    session_id: str = Depends(validate_session_id),
    limit: int = 50
):
    """Get chat history for a session."""
    try:
        history = await chat_agent_with_tools.get_chat_history(session_id, limit)
        
        # Convert to ChatMessage schema format
        messages = []
        for msg in history:
            message_data = schemas.ChatMessage(
                id=msg['id'],
                session_id=session_id,
                role=msg['role'],
                content=msg['content'],
                created_at=msg['created_at'],
                token_count=msg.get('token_count'),
                retrieved_chunks=msg.get('sources'),  # Include sources if available
                similarity_scores=None
            )
            messages.append(message_data)
        
        return messages
        
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chat history: {str(e)}"
        )


# System endpoints
@router.get("/system/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "openai": "configured",
            "pinecone": "configured"
        }
    }


@router.get("/system/stats")
async def get_system_stats(db: Session = Depends(get_database)):
    """Get system statistics."""
    try:
        # Document stats
        total_documents = db.query(Document).count()
        processed_documents = db.query(Document).filter(Document.is_processed == True).count()
        embedded_documents = db.query(Document).filter(Document.is_embedded == True).count()
        
        # Chat stats
        total_sessions = db.query(ChatSession).count()
        active_sessions = db.query(ChatSession).filter(ChatSession.is_active == True).count()
        
        # Research stats
        total_research_tasks = db.query(ResearchTask).count()
        completed_research_tasks = db.query(ResearchTask).filter(ResearchTask.status == "completed").count()
        
        # Pinecone stats
        try:
            pinecone_stats = embedding_service.get_index_stats()
        except Exception:
            pinecone_stats = {"error": "Could not retrieve Pinecone stats"}
        
        return {
            "documents": {
                "total": total_documents,
                "processed": processed_documents,
                "embedded": embedded_documents,
                "processing_rate": f"{processed_documents/max(1,total_documents)*100:.1f}%"
            },
            "chat": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions
            },
            "research": {
                "total_tasks": total_research_tasks,
                "completed_tasks": completed_research_tasks,
                "completion_rate": f"{completed_research_tasks/max(1,total_research_tasks)*100:.1f}%"
            },
            "vector_store": pinecone_stats,
            "system": {
                "uptime": "N/A",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system stats: {str(e)}"
        )


# Delete endpoints (for cleanup)
@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str = Depends(validate_document_id),
    db: Session = Depends(get_database)
):
    """Delete a document and all associated data."""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete from Pinecone
        await embedding_service.delete_document_embeddings(document_id)
        
        # Delete file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database (cascade will handle related records)
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/test-extraction/{document_id}")
async def test_metadata_extraction(
    document_id: str,
    db: Session = Depends(get_database)
):
    """
    Test endpoint to manually trigger metadata extraction for a specific document.
    Useful for debugging extraction issues.
    """
    try:
        logger.info(f"Testing metadata extraction for document {document_id}")
        
        # Check if document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Run metadata extraction
        result = await metadata_extractor.extract_metadata(document_id)
        
        logger.info(f"Test extraction completed for {document_id}")
        return {
            "document_id": document_id,
            "extraction_result": result,
            "message": "Metadata extraction test completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test extraction for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in test extraction: {str(e)}"
        )