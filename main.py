"""
FastAPI Backend for Multi-Tool Personal Assistant.
Provides REST API + SSE streaming for the chat interface.
"""

import os
import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from agent import run_agent_stream, run_agent_sync, get_llm
from research_agent import run_research_stream
from langchain_core.messages import SystemMessage, HumanMessage
from tools.pdf_tool import upload_and_index_pdf, get_pdf_metadata, clear_pdf_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Tool Assistant API",
    description="AI-powered assistant with Web Search, Calculator, PDF Q&A, and General Q&A",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

import history_store
import settings_store
import memory_store

# ─── Persistent chat history store ───────────────────────────────────────────
chat_sessions: dict[str, dict] = history_store.load_sessions()


# ─── Request/Response Models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    tools_used: Optional[list[str]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    chat_history: Optional[list[dict]] = None
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = True
    web_search: Optional[bool] = True
    speed_mode: Optional[str] = None  # 'fast' -> groq, 'slow' -> google, 'pro' -> mistral
    require_approval: Optional[bool] = False


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tools_used: list[str]
    timestamp: str


# ─── Health & Info Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Multi-Tool Assistant API",
        "version": "1.0.0",
        "status": "running",
        "tools": ["web_search", "calculator", "pdf_qa", "general_qa"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


class ApproveRequest(BaseModel):
    approved: bool


@app.post("/api/approve/{approval_id}")
async def approve_tool(approval_id: str, request: ApproveRequest):
    from agent import active_approvals
    if approval_id not in active_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found or expired")
    active_approvals[approval_id]["approved"] = request.approved
    active_approvals[approval_id]["event"].set()
    return {"status": "ok", "message": f"Tool execution {'approved' if request.approved else 'denied'}"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Supports both streaming (SSE) and non-streaming responses.
    """
    session_id = request.session_id or str(uuid.uuid4())

    # Get or create chat history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"messages": [], "created_at": datetime.now().isoformat()}

    # Use provided history or session history
    history = request.chat_history if request.chat_history else chat_sessions[session_id]["messages"]

    # Add user message to history
    user_msg = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat(),
    }
    chat_sessions[session_id]["messages"].append(user_msg)
    history_store.save_sessions(chat_sessions)

    # Resolve speed_mode -> provider override
    provider_override = None
    if request.speed_mode == "fast":
        provider_override = "groq"
        logger.info("Speed mode: FAST (Groq)")
    elif request.speed_mode == "slow":
        provider_override = "google"
        logger.info("Speed mode: SLOW (Google Gemini)")
    elif request.speed_mode == "pro":
        provider_override = "mistral"
        logger.info("Speed mode: PRO (Mistral)")

    if request.stream:
        # Return SSE stream
        return EventSourceResponse(
            _stream_response(request.message, history, request.temperature, session_id, request.web_search, provider_override, request.require_approval),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        start_time = datetime.now()
        result = await run_agent_sync(
            message=request.message,
            chat_history=history,
            temperature=request.temperature,
            enable_search=request.web_search,
            provider=provider_override,
        )
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Save assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": result["response"],
            "timestamp": datetime.now().isoformat(),
            "tools_used": result.get("tools_used", []),
            "latency_ms": latency_ms,
        }
        chat_sessions[session_id]["messages"].append(assistant_msg)
        history_store.save_sessions(chat_sessions)

        return JSONResponse(content={
            "session_id": session_id,
            "response": result["response"],
            "tools_used": result.get("tools_used", []),
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
        })


@app.post("/api/compare")
async def compare_models(request: ChatRequest):
    """
    Comparison endpoint. Runs the request through both Google (Gemini) and Groq
    synchronously and returns both responses and their latencies.
    """
    original_provider = settings.llm_provider
    
    google_response = ""
    latency_google = 0
    try:
        settings.llm_provider = "google"
        start_google = datetime.now()
        google_res = await run_agent_sync(message=request.message, chat_history=[], temperature=request.temperature)
        google_response = google_res["response"]
        latency_google = int((datetime.now() - start_google).total_seconds() * 1000)
    except Exception as e:
        logger.error(f"Error in google compare run: {e}")
        google_response = f"⚠️ Error querying Gemini: {str(e)}\n\nMake sure your Google Gemini API key is configured correctly in settings or environment."
        
    groq_response = ""
    latency_groq = 0
    try:
        settings.llm_provider = "groq"
        start_groq = datetime.now()
        groq_res = await run_agent_sync(message=request.message, chat_history=[], temperature=request.temperature)
        groq_response = groq_res["response"]
        latency_groq = int((datetime.now() - start_groq).total_seconds() * 1000)
    except Exception as e:
        logger.error(f"Error in groq compare run: {e}. Falling back to Gemini 2.5 Flash-lite.")
        try:
            settings.llm_provider = "google"
            orig_model = settings.gemini_model
            settings.gemini_model = "gemini-2.5-flash-lite"
            
            start_groq = datetime.now()
            groq_res = await run_agent_sync(message=request.message, chat_history=[], temperature=request.temperature)
            groq_response = f"*(Groq billing/API error: fell back to Gemini 2.5 Flash-lite)*\n\n{groq_res['response']}"
            latency_groq = int((datetime.now() - start_groq).total_seconds() * 1000)
            
            settings.gemini_model = orig_model
        except Exception as fallback_err:
            groq_response = f"⚠️ Error querying Groq: {str(e)}\n\nFallback also failed: {str(fallback_err)}"
        
    settings.llm_provider = original_provider
    
    return {
        "google": {
            "response": google_response,
            "latency_ms": latency_google,
        },
        "groq": {
            "response": groq_response,
            "latency_ms": latency_groq,
        }
    }


async def _stream_response(message: str, history: list, temperature: float, session_id: str, enable_search: bool = True, provider: Optional[str] = None, require_approval: bool = False):
    """Generator that yields SSE events from the agent."""
    full_response = ""
    tools_used = []
    start_time = datetime.now()

    async for event in run_agent_stream(message, history, temperature, enable_search, provider=provider, require_approval=require_approval):
        try:
            if event.get("type") == "token":
                full_response += event.get("content", "")
            elif event.get("type") == "tool_start":
                tools_used.append(event.get("tool", ""))

            if event.get("type") == "done":
                duration = (datetime.now() - start_time).total_seconds()
                event["latency_ms"] = int(duration * 1000)
                event["session_id"] = session_id

            yield {"data": json.dumps(event)}
        except Exception as e:
            logger.error(f"Error in stream response: {e}")
            yield {"data": json.dumps({"type": "error", "error": str(e)})}

    # Save assistant response to history
    if full_response:
        duration = (datetime.now() - start_time).total_seconds()
        assistant_msg = {
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat(),
            "tools_used": list(set(tools_used)),
            "latency_ms": int(duration * 1000),
        }
        if session_id in chat_sessions:
            chat_sessions[session_id]["messages"].append(assistant_msg)
            history_store.save_sessions(chat_sessions)


@app.post("/api/research")
async def deep_research(request: Request):
    """
    Stream the deep research agent process.
    """
    try:
        data = await request.json()
        topic = data.get("topic")
        if not topic:
            raise HTTPException(status_code=400, detail="Missing topic")
            
        async def event_generator():
            try:
                async for event in run_research_stream(topic):
                    yield event
            except Exception as e:
                logger.error(f"Research streaming error: {e}")
                yield {"type": "error", "error": str(e)}

        return StreamingResponse(
            (f"data: {json.dumps(e)}\n\n" async for e in event_generator()),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Research request parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── File Upload Endpoint ────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),
):
    """
    Upload a PDF file for indexing and Q&A.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Validate file size
    content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    # Save file temporarily
    session_id = session_id or str(uuid.uuid4())
    upload_path = Path(settings.upload_dir) / f"{session_id}_{file.filename}"

    try:
        with open(upload_path, "wb") as f:
            f.write(content)

        # Index the PDF
        result = upload_and_index_pdf(str(upload_path), file.filename)

        # Initialize session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"messages": [], "created_at": datetime.now().isoformat()}
            history_store.save_sessions(chat_sessions)

        return JSONResponse(content={
            "status": "success",
            "session_id": session_id,
            "file": {
                "name": file.filename,
                "size_mb": round(len(content) / (1024 * 1024), 2),
            },
            "indexing": result,
            "message": f"Successfully uploaded and indexed '{file.filename}'. You can now ask questions about this document.",
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        # Clean up uploaded file after indexing
        if upload_path.exists():
            upload_path.unlink()


# ─── Session Management ──────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions, sorted by most recently updated."""
    sessions = []
    for sid, data in chat_sessions.items():
        messages = data.get("messages", [])
        if not messages:
            continue  # Skip empty sessions — nothing to show in sidebar
        first_user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "New Chat")
        title = first_user_msg[:50] + ("…" if len(first_user_msg) > 50 else "")
        last_ts = messages[-1].get("timestamp", data.get("created_at", ""))
        sessions.append({
            "session_id": sid,
            "title": title,
            "message_count": len(messages),
            "created_at": data.get("created_at"),
            "last_updated": last_ts,
        })
    # Sort newest first
    sessions.sort(key=lambda s: s.get("last_updated") or "", reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session with full history."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": chat_sessions[session_id]["messages"],
        "created_at": chat_sessions[session_id].get("created_at"),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        history_store.save_sessions(chat_sessions)
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/sessions/{session_id}/clear-pdf")
async def clear_session_pdf(session_id: str):
    """Clear the indexed PDF data for a session."""
    clear_pdf_data()
    return {"status": "cleared", "message": "PDF data cleared"}


# ─── Memory Endpoints ────────────────────────────────────────────────────────

@app.get("/api/memories")
async def get_memories():
    """Retrieve all user memories."""
    memories = memory_store.load_memories()
    return {"memories": memories}


@app.delete("/api/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: str):
    """Delete a specific user memory."""
    success = memory_store.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id}


# ─── Settings Endpoints ──────────────────────────────────────────────────────

class AppSettings(BaseModel):
    llm_provider: str
    gemini_model: str
    mistral_model: str
    answering_method: str
    system_prompt: str
    github_token: str = ""
    slack_token: str = ""
    linkedin_token: str = ""
    apify_token: str = ""
    email_token: str = ""


@app.get("/api/settings")
async def get_app_settings():
    """Retrieve the current configuration settings."""
    return {
        "llm_provider": settings.llm_provider,
        "gemini_model": settings.gemini_model,
        "mistral_model": settings.mistral_model,
        "answering_method": settings.answering_method,
        "system_prompt": settings.system_prompt,
        "github_token": settings.github_token,
        "slack_token": settings.slack_token,
        "linkedin_token": settings.linkedin_token,
        "apify_token": settings.apify_token,
        "email_token": settings.email_token,
    }


@app.post("/api/settings")
async def update_app_settings(new_settings: AppSettings):
    """Update configuration settings and persist them to settings.json."""
    try:
        current_dict = settings_store.load_settings()
        
        current_dict["llm_provider"] = new_settings.llm_provider
        current_dict["gemini_model"] = new_settings.gemini_model
        current_dict["mistral_model"] = new_settings.mistral_model
        current_dict["answering_method"] = new_settings.answering_method
        current_dict["system_prompt"] = new_settings.system_prompt
        current_dict["github_token"] = new_settings.github_token
        current_dict["slack_token"] = new_settings.slack_token
        current_dict["linkedin_token"] = new_settings.linkedin_token
        current_dict["apify_token"] = new_settings.apify_token
        current_dict["email_token"] = new_settings.email_token
        
        settings_store.save_settings(current_dict)
        
        settings.llm_provider = new_settings.llm_provider
        settings.gemini_model = new_settings.gemini_model
        settings.mistral_model = new_settings.mistral_model
        settings.answering_method = new_settings.answering_method
        settings.system_prompt = new_settings.system_prompt
        settings.github_token = new_settings.github_token
        settings.slack_token = new_settings.slack_token
        settings.linkedin_token = new_settings.linkedin_token
        settings.apify_token = new_settings.apify_token
        settings.email_token = new_settings.email_token
        
        os.environ["LLM_PROVIDER"] = new_settings.llm_provider
        os.environ["GEMINI_MODEL"] = new_settings.gemini_model
        os.environ["MISTRAL_MODEL"] = new_settings.mistral_model
        os.environ["ANSWERING_METHOD"] = new_settings.answering_method
        os.environ["SYSTEM_PROMPT"] = new_settings.system_prompt
        os.environ["GITHUB_TOKEN"] = new_settings.github_token
        os.environ["SLACK_TOKEN"] = new_settings.slack_token
        os.environ["LINKEDIN_TOKEN"] = new_settings.linkedin_token
        os.environ["APIFY_TOKEN"] = new_settings.apify_token
        os.environ["EMAIL_TOKEN"] = new_settings.email_token
        
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Production Workspace Endpoints ──────────────────────────────────────────

class PlaygroundRequest(BaseModel):
    prompt: str
    system_prompt: str
    temperature: float
    max_tokens: int = 4096


@app.post("/api/playground")
async def playground_endpoint(request: PlaygroundRequest):
    """
    Direct developer playground query. Custom system prompt, temperature,
    and direct LLM completion return.
    """
    try:
        llm = get_llm(temperature=request.temperature)
        
        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))
        
        start_time = datetime.now()
        response = await llm.ainvoke(messages)
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Format the transaction log payload
        tx_request = {
            "model": getattr(llm, "model_name", getattr(llm, "model", "default")),
            "messages": [m.content if hasattr(m, "content") else str(m) for m in messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        tx_response = {
            "content": response.content,
            "latency_ms": latency_ms,
            "usage": response.response_metadata.get("token_usage", {})
        }
        
        return {
            "request_json": json.dumps(tx_request, indent=2),
            "response_json": json.dumps(tx_response, indent=2),
            "output": response.content,
            "latency_ms": latency_ms
        }
    except Exception as e:
        logger.error(f"Playground execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CouncilRequest(BaseModel):
    topic: str


@app.post("/api/council")
async def convene_council(request: CouncilRequest):
    """
    Multi-agent debate council. Runs a sequential debate on the user's topic
    between a Researcher, Grok, and Gemini.
    """
    try:
        # We will use Gemini to run all three parts with custom instructions
        llm = get_llm(temperature=0.7)
        
        # 1. Researcher Agent
        researcher_prompt = (
            "You are the Researcher Agent. Provide a objective, fact-based summary of the current "
            f"scientific and industry status regarding the topic: '{request.topic}'. Keep it to 3 sentences."
        )
        res_part = await llm.ainvoke([HumanMessage(content=researcher_prompt)])
        researcher_text = res_part.content
        
        # 2. Groq Agent
        grok_prompt = (
            "You are the Groq Agent, powered by Groq's ultra-fast LLM inference. Critique the following researcher summary about "
            f"'{request.topic}' in a witty, humorous, and highly critical tone. Keep it to 3 sentences.\n\n"
            f"Researcher Summary:\n{researcher_text}"
        )
        grok_part = await llm.ainvoke([HumanMessage(content=grok_prompt)])
        grok_text = grok_part.content
        
        # 3. Gemini Consensus Agent
        gemini_prompt = (
            "You are the Gemini Consensus Agent. Review the debate on "
            f"'{request.topic}' and synthesize a final balanced consensus recommendation based on the points below.\n\n"
            f"Researcher Point: {researcher_text}\n\n"
            f"Grok Critique: {grok_text}\n\n"
            "Keep the recommendation to 3 sentences."
        )
        gemini_part = await llm.ainvoke([HumanMessage(content=gemini_prompt)])
        gemini_text = gemini_part.content
        
        return {
            "researcher": researcher_text,
            "grok": grok_text,
            "gemini": gemini_text
        }
    except Exception as e:
        logger.error(f"Council debate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Code Generator Endpoint ─────────────────────────────────────────────────

class CodeGenRequest(BaseModel):
    description: str
    language: str = "Python"
    temperature: float = 0.2


@app.post("/api/code-gen")
async def code_gen_endpoint(request: CodeGenRequest):
    """
    Direct code generation endpoint. Takes a language + description and returns
    clean, production-ready code. Uses the active LLM provider (Groq or Gemini).
    """
    try:
        from tools.code_generator import build_code_prompt
        llm = get_llm(temperature=request.temperature)

        prompt = build_code_prompt(request.language, request.description)
        start_time = datetime.now()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        code = response.content.strip()
        # Strip accidental markdown fences
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return {
            "code": code,
            "language": request.language,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Route Test Endpoint ──────────────────────────────────────────────────────

class RouteTestRequest(BaseModel):
    query: str


@app.post("/api/route-test")
async def route_test(request: RouteTestRequest):
    """
    Classifies any user query into exactly one of three routes:
      - 'RAG'          : Questions about an uploaded document / file
      - 'Tool Calling' : Math, maps, weather, real-time data, calculations
      - 'LLM'          : General conversation, coding, explanations, anything else
    """
    query = request.query.strip()
    if not query:
        return {"route": "LLM", "confidence": 1.0, "reason": "Empty query — defaulting to LLM."}

    q = query.lower()

    # ── 1. RAG: uploaded document questions ───────────────────────
    rag_indicators = [
        "pdf", "document", "uploaded", "the doc", "the file",
        "read document", "faq", "contract", "clauses", "in the document",
        "according to", "from the pdf", "summary of the",
    ]
    if any(k in q for k in rag_indicators):
        return {
            "route": "RAG",
            "confidence": 0.97,
            "reason": "Query references an uploaded document or file — routed to RAG pipeline.",
        }

    # ── 2. Tool Calling: math, maps, weather, real-time data ─────────
    tool_indicators = [
        # Math / calculations
        "calculate", "compute", "equation", "algebra", "derivative",
        "integral", "factorial", "sum of", "multiply", "divide",
        "sqrt", "pow", "arithmetic",
        # Maps / location
        "map", "direction", "navigate", "distance from",
        "nearest", "near me", "coordinates", "route to",
        # Weather / real-time
        "weather", "temperature in", "forecast", "humidity", "raining",
        "sunny in", "climate in",
        # News / live data
        "latest news", "breaking news", "stock price", "crypto price",
        "exchange rate", "trending", "live score", "who won", "today's news",
    ]
    if any(k in q for k in tool_indicators):
        return {
            "route": "Tool Calling",
            "confidence": 0.96,
            "reason": "Query involves math, maps, weather, or real-time data — routed to Tool Calling.",
        }

    # Detect simple arithmetic expressions like "12 + 34" or "5 * 6"
    import re as _re
    if _re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', q):
        return {
            "route": "Tool Calling",
            "confidence": 0.98,
            "reason": "Arithmetic expression detected — routed to Tool Calling (Calculator).",
        }

    # ── 3. LLM: everything else ──────────────────────────────────
    llm_indicators = [
        "hello", "hi", "hey", "how are", "explain", "describe", "tell me",
        "how does", "how do", "how to", "why is", "why does", "why do",
        "what is", "what are", "what's", "when did", "when was", "where is",
        "can you", "could you", "help me", "i need", "i want", "please",
        "write", "create", "generate", "code", "function", "script",
        "translate", "summarize", "list", "give me", "show me", "draft",
        "compare", "difference", "pros and cons", "example of",
        "thanks", "thank you", "ok", "okay",
    ]
    if any(k in q for k in llm_indicators):
        return {
            "route": "LLM",
            "confidence": 0.95,
            "reason": "General conversation, coding, or knowledge query — routed to LLM.",
        }

    # Short messages default to LLM
    if len(q.split()) <= 5:
        return {
            "route": "LLM",
            "confidence": 0.88,
            "reason": "Short/ambiguous query — defaulting to LLM.",
        }

    # ── Semantic fallback (rare — tiny fast model) ──────────────────
    try:
        if settings.is_groq_configured:
            from langchain_groq import ChatGroq
            fast_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=settings.groq_api_key,
                temperature=0.0,
                max_tokens=60,
                max_retries=0,
            )
        elif settings.is_google_configured:
            from langchain_google_genai import ChatGoogleGenerativeAI
            fast_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=settings.google_api_key,
                temperature=0.0,
                max_output_tokens=60,
            )
        else:
            raise ValueError("No LLM provider configured for route testing.")
        sys_msg = (
            "Classify the user query into EXACTLY one of three routes:\n"
            "  'RAG'          — questions about an uploaded document or file\n"
            "  'Tool Calling' — math calculations, maps, weather, or real-time data\n"
            "  'LLM'          — everything else (chat, coding, explanations)\n\n"
            'Respond ONLY with JSON: {"route": "RAG"|"Tool Calling"|"LLM", '
            '"confidence": 0.0-1.0, "reason": "one sentence"}'
        )
        resp = await fast_llm.ainvoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=query[:300]),
        ])
        content = resp.content.strip().strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        data = json.loads(content)
        route = data.get("route", "LLM")
        # Normalise
        if route.lower() in ("rag", "pdf", "pdf_faq"):
            route = "RAG"
        elif route.lower() in ("tool calling", "tool_calling", "tools", "search", "math"):
            route = "Tool Calling"
        else:
            route = "LLM"
        return {
            "route": route,
            "confidence": float(data.get("confidence", 0.82)),
            "reason": data.get("reason", "Classified via semantic routing."),
        }
    except Exception as e:
        logger.error(f"Semantic route test failed: {e}.")
        return {
            "route": "LLM",
            "confidence": 0.80,
            "reason": "Could not classify precisely — defaulting to LLM.",
        }





# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
