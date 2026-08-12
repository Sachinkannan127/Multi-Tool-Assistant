"""
PDF Summarization & Q&A Tool using RAG with ChromaDB.
Extracts text from PDFs, chunks it, stores in vector DB, and answers questions.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

# Global state for current session's vector store
_vector_store = None
_pdf_metadata = {}


def _sanitize_text(text: str) -> str:
    """
    Sanitize extracted PDF text to prevent prompt injection.
    Removes control characters and suspicious patterns.
    """
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Remove common prompt injection patterns
    injection_patterns = [
        r'(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)',
        r'(?i)you\s+are\s+now\s+',
        r'(?i)forget\s+(everything|all)\s+',
        r'(?i)disregard\s+(all\s+)?(previous|prior)',
        r'(?i)system\s*:\s*',
        r'(?i)new\s+instructions?\s*:',
        r'(?i)override\s+(previous|prior|system)',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[REDACTED]', text)

    # Limit excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', '  ', text)

    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        doc.close()
        full_text = "\n\n".join(text_parts)
        return _sanitize_text(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def get_embeddings():
    """Get Google Gemini embeddings model, or fallback to fake embeddings if not configured."""
    if not settings.is_google_configured:
        logger.warning("Using FakeEmbeddings because GOOGLE_API_KEY is not configured or is a placeholder.")
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=768)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    model_name = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )


def upload_and_index_pdf(file_path: str, filename: str) -> dict:
    """
    Upload a PDF, extract text, chunk it, and store in ChromaDB.
    Returns metadata about the indexed document.
    """
    global _vector_store, _pdf_metadata

    # Validate file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max {settings.max_upload_size_mb}MB)")

    if not filename.lower().endswith('.pdf'):
        raise ValueError(f"Invalid file type. Only PDF files are allowed.")

    # Extract text
    text = extract_text_from_pdf(file_path)
    if len(text.strip()) < 50:
        raise ValueError("PDF appears to contain very little text. It may be a scanned image PDF.")

    # Create document hash for deduplication
    doc_hash = hashlib.md5(text.encode()).hexdigest()

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_text(text)

    # Create metadata for each chunk
    metadatas = [
        {"source": filename, "chunk_index": i, "doc_hash": doc_hash}
        for i in range(len(chunks))
    ]

    # Store in ChromaDB
    try:
        from langchain_chroma import Chroma

        embeddings = get_embeddings()

        # Use a collection name based on document hash
        collection_name = f"pdf_{doc_hash[:12]}"

        # Create or get the vector store
        _vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

        # Check if already indexed
        existing = _vector_store.get()
        if not existing or len(existing.get("ids", [])) == 0:
            # Add documents
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            _vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=ids,
            )

        _pdf_metadata = {
            "filename": filename,
            "num_chunks": len(chunks),
            "text_length": len(text),
            "doc_hash": doc_hash,
            "collection_name": collection_name,
        }

        return {
            "status": "success",
            "filename": filename,
            "pages_estimated": text.count("--- Page") or 1,
            "chunks_created": len(chunks),
            "characters_extracted": len(text),
        }

    except Exception as e:
        logger.error(f"Error indexing PDF: {e}")
        raise ValueError(f"Failed to index PDF: {str(e)}")


def create_pdf_tool():
    """Create the PDF Q&A tool."""

    @tool
    def pdf_qa_tool(query: str) -> str:
        """
        Answer questions based on the uploaded PDF document using RAG.
        Use this tool when the user asks questions about an uploaded PDF,
        requests a summary of the PDF, or wants information from the document.

        Input: A question or query about the uploaded PDF document.
        """
        global _vector_store, _pdf_metadata

        if _vector_store is None:
            return "No PDF has been uploaded yet. Please upload a PDF first before asking questions about it."

        try:
            # Retrieve relevant chunks
            retriever = _vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
            docs = retriever.invoke(query)

            if not docs:
                return "I couldn't find relevant information in the uploaded PDF to answer your question."

            # Format context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                context_parts.append(f"[Excerpt {i}]:\n{content}")

            context = "\n\n---\n\n".join(context_parts)

            filename = _pdf_metadata.get("filename", "uploaded document")

            return (
                f"Relevant excerpts from '{filename}':\n\n"
                f"{context}\n\n"
                f"---\nUse these excerpts to answer the user's question: {query}"
            )

        except Exception as e:
            logger.error(f"Error in PDF Q&A: {e}")
            return f"Error searching PDF: {str(e)}"

    return pdf_qa_tool


def get_pdf_metadata() -> dict:
    """Get metadata about the currently indexed PDF."""
    return _pdf_metadata.copy()


def clear_pdf_data():
    """Clear the current PDF data from memory."""
    global _vector_store, _pdf_metadata
    _vector_store = None
    _pdf_metadata = {}
