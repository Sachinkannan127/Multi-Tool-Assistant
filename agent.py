"""
LangChain Agent orchestration using ReAct framework.
Integrates all tools: Web Search, Calculator, PDF Q&A, and General Q&A.
"""

import os
import json
import logging
import asyncio
import uuid
import contextvars
from typing import AsyncGenerator, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import AsyncCallbackHandler

# Context variables to pass stream queue and settings to tools
sse_queue_var = contextvars.ContextVar("sse_queue", default=None)
require_approval_var = contextvars.ContextVar("require_approval", default=False)
active_approvals: dict[str, dict] = {}

from config import settings
from tools.calculator import calculator_tool
from tools.search import web_search_tool
from tools.pdf_tool import create_pdf_tool, get_pdf_metadata
from tools.memory import save_memory_tool, list_memories_tool, delete_memory_tool
from tools.scraper import web_scraper_tool
from tools.code_generator import code_generator_tool
from tools.youtube_tool import youtube_tool
from tools.instagram_tool import instagram_tool
from tools.github_tool import github_tool
from tools.slack_tool import slack_tool
from tools.wikipedia_tool import wikipedia_tool
import memory_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a powerful Multi-Tool AI Assistant. You help users by intelligently selecting and using the right tools for their needs.

## Decision Framework:
1. **web_search_tool** - Search the internet for real-time information, news, weather, current events, and up-to-date facts.
2. **calculator_tool** - Perform mathematical calculations, unit conversions, and financial calculations.
3. **pdf_qa_tool** - Answer questions about uploaded PDF documents using retrieved context.
4. **web_scraper_tool** - Fetch and extract full text content and all hyperlinks from a specific URL provided by the user (general internet data fetching).
5. **code_generator_tool** - Generate clean, production-ready code in ANY programming language from a natural-language description.
6. **youtube_tool** - Extract transcripts from YouTube videos to summarize or analyze them.
7. **instagram_tool** - Extract data from Instagram posts or profiles to summarize them.
8. **wikipedia_tool** - Search Wikipedia and retrieve article summaries or content.

You have access to various tools for web search, scraping, code generation, YouTube, Instagram, Wikipedia, GitHub, etc. Use them appropriately based on the user's request.

## Decision Framework:
- If the user asks a **general question** or asks to **search/fetch data from the internet, YouTube, or Instagram** without providing a specific URL → use `web_search_tool` to find relevant links or information first.
- If the user asks about **current events, news, weather, prices, or real-time data** → use `web_search_tool`
- If the user asks for **math calculations, conversions, or financial computations** → use `calculator_tool`
- If the user asks about an **uploaded PDF or document content** → use `pdf_qa_tool`
- If the user provides a **specific URL** and asks to read, summarize, scrape, extract data, or list links from it → use `web_scraper_tool`
- If the user provides a **YouTube URL** and asks to summarize or fetch data → use `youtube_tool`
- If the user provides an **Instagram URL** and asks to summarize or fetch data → use `instagram_tool`
- If the user asks for factual information, background on a topic, or explicitly mentions Wikipedia → use `wikipedia_tool`
- If the user asks to **write, create, generate, build, or implement code** in any language → use `code_generator_tool` with input format "Language: description"
- If the user asks about **GitHub repositories, issues, or code** → use `github_tool`
- If the user asks to **send a message on Slack** → use `slack_tool`
- If the question is **general knowledge** that you can answer from training data → answer directly

## Response Guidelines:
- You must ALWAYS structure your response in this exact order:
  1. **Summary**: Provide a brief, high-level summary of the answer first.
  2. **Main Content**: Provide the detailed, comprehensive response using structured markdown headings (##, ###), bullet points, and bold text.
  3. **Sources**: End your response with a "Sources:" section listing the direct URLs or references of any tools you used.
- Provide rich, detailed, and enhanced answers that fully address the user's intent. 
- If the user asks to "differentiate", "compare", or "differences between", ALWAYS output the comparison in a Markdown Table format.
- EXTREMELY IMPORTANT: Use emojis heavily in your output text format! Prefix every heading, sub-heading, and bullet point with a relevant emoji (e.g., 🌟, 📊, ✅, etc.) to make the text highly engaging and visually appealing.
- When using web search, include source citations with URLs
- When using the calculator, show the expression and result clearly
- When answering from PDF, reference specific parts of the document
- When using web_scraper_tool, present the extracted content cleanly and list links as a numbered or bulleted list
- When generating code, present it in a clean code block with the language specified
- Be conversational and helpful
- Use markdown formatting for readability

## Important:
- **SUMMARIZATION & ANTI-HALLUCINATION:** When summarizing or extracting data from tools (web search, YouTube, Instagram, web scraper, PDF), you MUST synthesize the information in your own words. DO NOT just copy and paste the exact text verbatim. However, you MUST ONLY use the facts provided by the tool output. DO NOT invent, hallucinate, or assume any information that is not explicitly present. If the tool does not provide the information, state clearly that you do not have it.
- For multi-step problems, break them down and use multiple tools as needed
- Always verify calculations when precision matters
- Provide context for your answers, not just raw data
- Never call web_scraper_tool for a general query — only when a specific URL is provided
- Always use code_generator_tool when the user wants actual code — never write code in plain text responses
"""


def _get_system_prompt() -> str:
    """Retrieve the base system prompt and inject active user memories/facts."""
    base_prompt = os.getenv("SYSTEM_PROMPT") or settings.system_prompt or SYSTEM_PROMPT
    
    try:
        memories = memory_store.load_memories()
        if memories:
            memories_str = "\n".join([f"- {m['content']} (ID: {m['id']})" for m in memories])
        else:
            memories_str = "No current memories saved."
    except Exception as e:
        logger.error(f"Error loading memories for system prompt: {e}")
        memories_str = "Error loading memories."
        
    memory_section = f"""

## Current User Memories & Facts:
These are the details you remember about the user across chat sessions:
{memories_str}

Guidelines for managing memories:
- If the user shares new personal facts, preferences, background, configurations, or instructions that should persist across sessions, use the `save_memory_tool` to save them.
- If a memory is no longer true, incorrect, or has been updated, use the `delete_memory_tool` to remove it by its ID.
- DO NOT explicitly list or repeat these memories to the user in your response unless they ask you to, or if it naturally fits the conversation. Use them implicitly to guide your responses.
"""
    return base_prompt + "\n" + memory_section


def get_llm(temperature: float = 0.1, provider: Optional[str] = None):
    """Get the LLM instance based on configuration.
    
    Args:
        temperature: Sampling temperature.
        provider: Optional override — 'groq' (fast) or 'google' (slow/capable).
                  If None, falls back to settings.llm_provider.
    """
    active = provider or settings.llm_provider
    if active in ["google", "gemini"]:
        model_name = os.getenv("GEMINI_MODEL", settings.gemini_model)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            streaming=True,
            max_output_tokens=8192,
        )
    elif active == "mistral":
        model_name = os.getenv("MISTRAL_MODEL", settings.mistral_model)
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model_name,
            mistral_api_key=settings.mistral_api_key,
            temperature=temperature,
            max_tokens=4096,
            max_retries=0,
        )
    else:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
        return ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
            streaming=True,
            max_tokens=4096,
            max_retries=0,          # Fail fast — we handle retries via Gemini fallback
        )


def _is_rate_limit_error(e: Exception) -> bool:
    """Return True if the exception is a Groq / OpenAI rate-limit (429) error."""
    msg = str(e).lower()
    return (
        "rate_limit" in msg
        or "rate limit" in msg
        or "429" in msg
        or "too many requests" in msg
        or "tokens per minute" in msg
        or "requests per minute" in msg
    )




def wrap_tool_with_approval(tool):
    """Wrap a tool's async execute function to support tool call approvals."""
    import functools
    orig_arun = getattr(tool, "_arun", None)
    orig_run = getattr(tool, "_run", None)

    if orig_arun:
        @functools.wraps(orig_arun)
        async def wrapped_arun(*args, **kwargs):
            req_approval = require_approval_var.get()
            queue = sse_queue_var.get()

            if req_approval and queue:
                approval_id = str(uuid.uuid4())
                event = asyncio.Event()

                tool_input = ""
                if args:
                    tool_input = str(args[0])
                elif kwargs:
                    # Filter out system arguments for cleaner display
                    display_kwargs = {k: v for k, v in kwargs.items() if k not in ["config", "run_manager"]}
                    tool_input = json.dumps(display_kwargs)

                active_approvals[approval_id] = {
                    "event": event,
                    "approved": None,
                    "tool": tool.name,
                    "input": tool_input,
                }

                # Emit approval request SSE event
                await queue.put({
                    "type": "tool_approval_request",
                    "approval_id": approval_id,
                    "tool": tool.name,
                    "emoji": getattr(tool, "emoji", "⚙️") if hasattr(tool, "emoji") else "⚙️",
                    "label": f"Approval requested for {tool.name}",
                    "input": tool_input,
                })

                # Block until event is set
                await event.wait()

                decision = active_approvals[approval_id]["approved"]
                active_approvals.pop(approval_id, None)

                if not decision:
                    return f"Tool execution rejected by the user."

            return await orig_arun(*args, **kwargs)

        tool._arun = wrapped_arun

    if orig_run:
        @functools.wraps(orig_run)
        def wrapped_run(*args, **kwargs):
            return orig_run(*args, **kwargs)
        tool._run = wrapped_run

    return tool


def get_tools(enable_search: bool = True):
    """Return the list of active tools based on settings."""
    tools = [
        calculator_tool,
        create_pdf_tool(),
        save_memory_tool,
        list_memories_tool,
        delete_memory_tool,
        web_scraper_tool,
        code_generator_tool,
        youtube_tool,
        instagram_tool,
        wikipedia_tool,
    ]
    
    if settings.github_token:
        tools.append(github_tool)
    if settings.slack_token:
        tools.append(slack_tool)
        
    if enable_search and settings.is_tavily_configured:
        tools.insert(0, web_search_tool)
        
    return tools


# ─── Agent executor ────────────────────────────────────────────────────────────

def create_agent_executor(temperature: float = 0.1, enable_search: bool = True, provider: Optional[str] = None):
    """Return a new LangGraph AgentExecutor."""
    llm = get_llm(temperature=temperature, provider=provider)
    tools = get_tools(enable_search=enable_search)
    sys_prompt = _get_system_prompt()

    from langgraph.prebuilt import create_react_agent
    import inspect
    sig = inspect.signature(create_react_agent)
    kwargs = {}
    if "prompt" in sig.parameters:
        kwargs["prompt"] = sys_prompt
    elif "state_modifier" in sig.parameters:
        kwargs["state_modifier"] = sys_prompt
    elif "messages_modifier" in sig.parameters:
        kwargs["messages_modifier"] = sys_prompt
    else:
        kwargs["state_modifier"] = sys_prompt
    return create_react_agent(llm, tools, **kwargs)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming tool usage events."""

    def __init__(self):
        self.events: list[dict] = []
        self.current_tool: Optional[str] = None

    async def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.current_tool = tool_name
        self.events.append({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str[:500],
        })

    async def on_tool_end(self, output, **kwargs):
        self.events.append({
            "type": "tool_end",
            "tool": self.current_tool,
            "output": str(output)[:1000],
        })
        self.current_tool = None

    async def on_tool_error(self, error, **kwargs):
        self.events.append({
            "type": "tool_error",
            "tool": self.current_tool,
            "error": str(error),
        })

    async def on_llm_new_token(self, token: str, **kwargs):
        self.events.append({
            "type": "token",
            "content": token,
        })


async def _determine_chat_route(message: str, has_pdf: bool) -> str:
    """
    Fast heuristic router — covers 95%+ of queries in 0ms.
    Falls back to a lightweight LLM call (8b model) only for truly ambiguous inputs.
    """
    query = message.strip().lower()

    # ── 0. Memory operations (tool calling) ──────────────────────────
    memory_indicators = ["remember", "forget", "save this", "note that", "my name is", "i am a", "i prefer", "don't forget"]
    if any(k in query for k in memory_indicators):
        return "tool_calling"

    # ── 1. Math (tool calling) ────────────────────────────────────────
    math_indicators = ["+", "*", "%", "calculate", "equation", "sum", "multiply", "divide", "subtract", "algebra", "pow", "sqrt", "factorial", "derivative", "integral", "convert"]
    if any(k in query for k in math_indicators) or (query.replace(" ", "").replace(".","").isdigit() and len(query) > 0):
        return "tool_calling"
    # Simple arithmetic expression like "12 + 34" or "5 * 6"
    import re as _re
    if _re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', query):
        return "tool_calling"

    # ── 2. PDF (pdf_faq) ─────────────────────────────────────────────
    pdf_indicators = ["pdf", "document", "uploaded file", "read document", "faq", "uploaded", "summary of the document", "contract", "clauses", "the file", "what does the doc"]
    if has_pdf and any(k in query for k in pdf_indicators):
        return "pdf_faq"

    # ── 3. Web search & Scrapers (tool calling) ─────────────────────────────────
    search_indicators = [
        "search", "google", "news", "weather", "today", "current", "latest",
        "stock", "price of", "live", "real-time", "right now", "this week",
        "map", "location", "temperature in", "weather in", "score",
        "who won", "what happened", "trending", "youtube", "instagram", "summarize website",
        "github", "slack", "wikipedia"
    ]
    if any(k in query for k in search_indicators):
        return "tool_calling"

    # ── 4. General conversation → direct_llm (NO extra LLM call) ─────
    direct_indicators = [
        "hello", "hi", "hey", "how are", "what is", "what are", "what's",
        "explain", "describe", "tell me", "how does", "how do", "how to",
        "why is", "why does", "why do", "when did", "when was", "where is",
        "can you", "could you", "please", "help me", "i need", "i want",
        "write", "create", "generate", "make", "build", "code", "function",
        "translate", "summarize", "list", "give me", "show me", "draft",
        "compare", "difference between", "pros and cons", "example of",
        "thanks", "thank you", "ok", "okay", "great", "cool", "nice",
    ]
    if any(k in query for k in direct_indicators):
        return "direct_llm"

    # ── 5. Short messages → direct_llm ────────────────────────────────
    if len(query.split()) <= 6:
        return "direct_llm"

    # ── 6. Semantic fallback (rare — use smallest/fastest model) ──────
    try:
        # Use a tiny 8b model just for routing — not the main model
        fast_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=20,
            max_retries=0,
        )
        system_instruction = (
            f"Route to ONE of: pdf_faq (has_pdf={has_pdf}), tool_calling, direct_llm. "
            "Return ONLY a JSON: {\"route\": \"...\"}  No explanation."
        )
        resp = await fast_llm.ainvoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=query[:200]),
        ])
        content = resp.content.strip().strip("`")
        if content.startswith("json"): content = content[4:]
        route = json.loads(content).get("route", "direct_llm")
        if route == "pdf_faq" and not has_pdf:
            return "direct_llm"
        return route
    except Exception as e:
        logger.error(f"Semantic routing failed: {e}. Defaulting to direct_llm.")
        return "direct_llm"



async def run_agent_stream(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    enable_search: bool = True,
    provider: Optional[str] = None,
    require_approval: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Run the agent and yield SSE-formatted events.
    Routes to different answering methods based on configuration.
    
    Args:
        provider: Optional override — 'groq' (fast mode) or 'google' (slow/capable mode).
    """
    if "pro" in message.lower().split():
        provider = "mistral"

    method = settings.answering_method.lower()
    
    if method == "pdf_faq":
        async for event in _run_pdf_faq_stream(message, chat_history, temperature, provider=provider):
            yield event
    elif method == "direct_llm":
        async for event in _run_direct_llm_stream(message, chat_history, temperature, provider=provider):
            yield event
    else:  # tool_calling (default, dynamic auto-route)
        has_pdf = get_pdf_metadata() is not None
        route = await _determine_chat_route(message, has_pdf)
        
        logger.info(f"Dynamic router selected route: '{route}' for query: '{message}'")
        
        if route == "pdf_faq":
            async for event in _run_pdf_faq_stream(message, chat_history, temperature, provider=provider):
                yield event
        elif route == "direct_llm":
            async for event in _run_direct_llm_stream(message, chat_history, temperature, provider=provider):
                yield event
        else:
            async for event in _run_tool_calling_stream(message, chat_history, temperature, enable_search, provider=provider, require_approval=require_approval):
                yield event


async def _run_tool_calling_stream(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    enable_search: bool = True,
    provider: Optional[str] = None,
    require_approval: bool = False,
) -> AsyncGenerator[dict, None]:
    """Run the ReAct tool-calling agent for streaming with approval support."""
    if not settings.is_llm_configured:
        async for event in _run_agent_stream_mock(message, chat_history, require_approval=require_approval):
            yield event
        return

    # Create queue for merging events from astream_events and wrapped tool approvals
    queue = asyncio.Queue()

    # Set context variables in this async context
    sse_queue_var.set(queue)
    require_approval_var.set(require_approval)

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        for msg in chat_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        executor = create_agent_executor(temperature=temperature, enable_search=enable_search, provider=provider)

        # Add PDF context if available
        pdf_meta = get_pdf_metadata()
        if pdf_meta:
            message = f"[PDF '{pdf_meta.get('filename', 'document')} is available for reference]\n\n{message}"
        
        messages.append(HumanMessage(content=message))

        # Map tool names to user-friendly indicators
        tool_indicators = {
            "web_search_tool":       {"emoji": "🔍", "label": "Searching the web..."},
            "calculator_tool":  {"emoji": "🧮", "label": "Calculating..."},
            "pdf_qa_tool":      {"emoji": "📄", "label": "Reading PDF..."},
            "web_scraper_tool": {"emoji": "🕷️", "label": "Scraping webpage..."},
            "youtube_tool":     {"emoji": "📺", "label": "Extracting YouTube..."},
            "instagram_tool":   {"emoji": "📸", "label": "Scraping Instagram..."},
            "wikipedia_tool":   {"emoji": "🏛️", "label": "Searching Wikipedia..."},
        }

        # Sub-task to run the agent in the same context context
        async def run_agent_executor():
            try:
                async for event in executor.astream_events(
                    {"messages": messages},
                    version="v2"
                ):
                    kind = event.get("event")
                    name = event.get("name")

                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            await queue.put(_sse_event({
                                "type": "token",
                                "content": content,
                            }))

                    elif kind == "on_tool_start":
                        tool_name = name
                        tool_input = event["data"].get("input")
                        indicator = tool_indicators.get(tool_name, {"emoji": "⚙️", "label": f"Using {tool_name}..."})
                        await queue.put(_sse_event({
                            "type": "tool_start",
                            "tool": tool_name,
                            "emoji": indicator["emoji"],
                            "label": indicator["label"],
                            "input": str(tool_input)[:200] if tool_input else "",
                        }))

                    elif kind == "on_tool_end":
                        tool_name = name
                        tool_output = event["data"].get("output")
                        await queue.put(_sse_event({
                            "type": "tool_end",
                            "tool": tool_name,
                            "output": str(tool_output) if tool_output is not None else "",
                        }))

                await queue.put(_sse_event({"type": "done"}))
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(None)

        # Create task (inherits context variables in Python 3.7+)
        task = asyncio.create_task(run_agent_executor())

        # Consume from queue and yield
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        await task

    except Exception as e:
        active_provider = provider or settings.llm_provider
        logger.error(f"Agent stream error with provider {active_provider}: {e}")
        yield _sse_event({
            "type": "error",
            "error": str(e),
        })


async def _run_pdf_faq_stream(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    provider: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run PDF FAQ mode: direct question-answering from uploaded PDF only."""
    try:
        pdf_meta = get_pdf_metadata()
        
        if not pdf_meta:
            # No PDF uploaded, fall back to direct LLM
            async for event in _run_direct_llm_stream(message, chat_history, temperature):
                yield event
            return

        # Use the PDF tool to retrieve context
        from tools.pdf_tool import create_pdf_tool
        pdf_qa = create_pdf_tool()
        
        yield _sse_event({
            "type": "tool_start",
            "tool": "pdf_qa_tool",
            "emoji": "📄",
            "label": "Searching document...",
        })
        
        await asyncio.sleep(0.3)
        
        # Get the context from PDF
        pdf_context = pdf_qa.invoke(message)
        
        yield _sse_event({
            "type": "tool_end",
            "tool": "pdf_qa_tool",
        })
        
        # Generate response using LLM with PDF context
        llm = get_llm(temperature=temperature)
        
        system_msg = f"""You are a helpful assistant answering questions about a document.
Use the provided document excerpts to answer the user's question accurately.
If you cannot find the answer in the document, say so explicitly.
IMPORTANT: When summarizing or extracting data from the document, you MUST synthesize the information in your own words. DO NOT just copy and paste the exact text verbatim.

You must ALWAYS structure your response in this exact order:
1. **Summary**: Provide a brief, high-level summary of the answer first.
2. **Main Content**: Provide the detailed, comprehensive response using structured markdown headings (##, ###), bullet points, and bold text.
3. **Sources**: End your response with a "Sources:" section listing the direct URLs or references of any tools you used (or simply state 'No external sources used' if none).

Document: {pdf_meta.get('filename', 'uploaded document')}
"""
        
        final_prompt = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Document context:\n{pdf_context}\n\nQuestion: {message}"),
        ]
        
        # Stream the response token by token
        async for chunk in llm.astream(final_prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield _sse_event({
                    "type": "token",
                    "content": content,
                })
        
        yield _sse_event({"type": "done"})
        
    except Exception as e:
        logger.error(f"PDF FAQ error: {e}", exc_info=True)
        yield _sse_event({
            "type": "error",
            "error": str(e),
        })


async def _run_direct_llm_stream(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    provider: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run direct LLM mode: generate responses without tools."""
    try:
        if not settings.is_llm_configured:
            # Fall back to mock response
            async for event in _run_agent_stream_mock(message, chat_history):
                yield event
            return
        
        llm = get_llm(temperature=temperature, provider=provider)
        
        # Build LangChain message format
        messages = [
            SystemMessage(content="""You are a helpful multi-tool personal assistant.
Provide clear, accurate, and helpful answers. Be conversational and friendly.

You must ALWAYS structure your response in this exact order:
1. **Summary**: Provide a brief, high-level summary of the answer first.
2. **Main Content**: Provide the detailed, comprehensive response using structured markdown headings (##, ###), bullet points, and bold text.
3. **Sources**: End your response with a "Sources:" section listing the direct URLs or references of any tools you used (or simply state 'No external sources used' if none).

If the user asks to "differentiate", "compare", or asks for the "differences between" things, ALWAYS output the comparison in a Markdown Table format.
EXTREMELY IMPORTANT: Use emojis heavily in your output text format! Prefix every heading, sub-heading, and bullet point with a relevant emoji (e.g., 🌟, 📊, ✅, etc.) to make the text highly engaging and visually appealing.
If you need current information (weather, news, prices) or need to perform calculations, 
mention that you could use tools for that, but provide your best answer based on your knowledge."""),
        ]
        
        # Add recent chat history for context
        for msg in chat_history[-4:]:  # Last 4 messages for context
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Stream the response
        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield _sse_event({
                    "type": "token",
                    "content": content,
                })
        
        yield _sse_event({"type": "done"})
        
    except Exception as e:
        logger.error(f"Direct LLM error: {e}", exc_info=True)
        yield _sse_event({
            "type": "error",
            "error": str(e),
        })


def _build_final_prompt(
    user_message: str,
    intermediate_steps: list,
    chat_history: list,
    pdf_meta: dict,
) -> list:
    """Build the final prompt for generating the streamed response."""
    sys_prompt = _get_system_prompt()
    messages = [SystemMessage(content=sys_prompt)]

    # Add chat history
    for msg in chat_history[-6:]:  # Last 6 messages for context
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Add tool results as context
    if intermediate_steps:
        tool_context = "\n\n## Tool Results:\n"
        for action, observation in intermediate_steps:
            tool_name = action.tool if hasattr(action, 'tool') else str(action)
            tool_context += f"\n### {tool_name}:\n{str(observation)[:2000]}\n"

        user_msg_with_context = f"{user_message}\n{tool_context}\n\nPlease synthesize a helpful response using the tool results above."
    else:
        user_msg_with_context = user_message

    if pdf_meta:
        user_msg_with_context = f"[PDF '{pdf_meta.get('filename', 'document')}' is available]\n\n{user_msg_with_context}"

    messages.append(HumanMessage(content=user_msg_with_context))
    return messages


def _sse_event(data: dict) -> dict:
    """Return the event data dictionary directly (formatting happens at the handler level)."""
    return data


async def run_agent_sync(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    enable_search: bool = True,
    provider: Optional[str] = None,
) -> dict:
    """
    Run the agent synchronously and return the full result.
    Routes to different answering methods based on configuration.
    """
    if "pro" in message.lower().split():
        provider = "mistral"

    method = settings.answering_method.lower()
    
    if method == "pdf_faq":
        return await _run_pdf_faq_sync(message, chat_history, temperature, provider=provider)
    elif method == "direct_llm":
        return await _run_direct_llm_sync(message, chat_history, temperature, provider=provider)
    else:  # tool_calling (default, dynamic auto-route)
        has_pdf = get_pdf_metadata() is not None
        route = await _determine_chat_route(message, has_pdf)
        
        logger.info(f"Dynamic router selected route: '{route}' for query: '{message}'")
        
        if route == "pdf_faq":
            return await _run_pdf_faq_sync(message, chat_history, temperature, provider=provider)
        elif route == "direct_llm":
            return await _run_direct_llm_sync(message, chat_history, temperature, provider=provider)
        else:
            return await _run_tool_calling_sync(message, chat_history, temperature, enable_search, provider=provider)


async def _run_tool_calling_sync(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    enable_search: bool = True,
    provider: Optional[str] = None,
) -> dict:
    """Run the ReAct tool-calling agent synchronously."""
    if not settings.is_llm_configured:
        return _run_agent_sync_mock(message, chat_history)

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        for msg in chat_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        executor = create_agent_executor(temperature=temperature, enable_search=enable_search, provider=provider)

        pdf_meta = get_pdf_metadata()
        if pdf_meta:
            message = f"[PDF '{pdf_meta.get('filename', 'document')} is available for reference]\n\n{message}"

        messages.append(HumanMessage(content=message))

        result = await executor.ainvoke(
            {"messages": messages}
        )

        # Extract output from the last message
        output_msg = result["messages"][-1].content
        
        # Tools used can be parsed from the state, but simple for now
        tools_used = []
        for m in result["messages"]:
            if hasattr(m, 'name') and m.type == 'tool':
                tools_used.append(m.name)

        return {
            "response": output_msg,
            "tools_used": tools_used,
            "intermediate_steps": len(tools_used),
        }

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {
            "response": f"I encountered an error: {str(e)}",
            "tools_used": [],
            "error": str(e),
        }


async def _run_pdf_faq_sync(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    provider: Optional[str] = None,
) -> dict:
    """Run PDF FAQ mode synchronously."""
    try:
        pdf_meta = get_pdf_metadata()
        
        if not pdf_meta:
            return await _run_direct_llm_sync(message, chat_history, temperature, provider=provider)
        
        from tools.pdf_tool import create_pdf_tool
        pdf_qa = create_pdf_tool()
        pdf_context = pdf_qa.invoke(message)
        
        llm = get_llm(temperature=temperature, provider=provider)
        
        system_msg = f"""You are a helpful assistant answering questions about a document.
Use the provided document excerpts to answer the user's question accurately.
If you cannot find the answer in the document, say so explicitly.
IMPORTANT: When summarizing or extracting data from the document, you MUST synthesize the information in your own words. DO NOT just copy and paste the exact text verbatim.

You must ALWAYS structure your response in this exact order:
1. **Summary**: Provide a brief, high-level summary of the answer first.
2. **Main Content**: Provide the detailed, comprehensive response using structured markdown headings (##, ###), bullet points, and bold text.
3. **Sources**: End your response with a "Sources:" section listing the direct URLs or references of any tools you used (or simply state 'No external sources used' if none).

Document: {pdf_meta.get('filename', 'uploaded document')}
"""
        
        final_prompt = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Document context:\n{pdf_context}\n\nQuestion: {message}"),
        ]
        
        # Invoke the model
        response = await llm.ainvoke(final_prompt)
        
        return {
            "response": response.content if hasattr(response, 'content') else str(response),
            "tools_used": ["pdf_qa_tool"],
            "mode": "pdf_faq",
        }
        
    except Exception as e:
        logger.error(f"PDF FAQ error: {e}", exc_info=True)
        return {
            "response": f"Error querying PDF: {str(e)}",
            "tools_used": [],
            "error": str(e),
        }


async def _run_direct_llm_sync(
    message: str,
    chat_history: list[dict],
    temperature: float = 0.1,
    provider: Optional[str] = None,
) -> dict:
    """Run direct LLM mode synchronously."""
    try:
        if not settings.is_llm_configured:
            return _run_agent_sync_mock(message, chat_history)
        
        llm = get_llm(temperature=temperature, provider=provider)
        
        # Build LangChain message format
        messages = [
            SystemMessage(content="""You are a helpful multi-tool personal assistant.
Provide clear, accurate, and helpful answers. Be conversational and friendly.

You must ALWAYS structure your response in this exact order:
1. **Summary**: Provide a brief, high-level summary of the answer first.
2. **Main Content**: Provide the detailed, comprehensive response using structured markdown headings (##, ###), bullet points, and bold text.
3. **Sources**: End your response with a "Sources:" section listing the direct URLs or references of any tools you used (or simply state 'No external sources used' if none).

If you need current information (weather, news, prices) or need to perform calculations,
mention that you could use tools for that, but provide your best answer based on your knowledge."""),
        ]
        
        # Add recent chat history for context
        for msg in chat_history[-4:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Invoke the model
        response = await llm.ainvoke(messages)
        
        return {
            "response": response.content if hasattr(response, 'content') else str(response),
            "tools_used": [],
            "mode": "direct_llm",
        }
        
    except Exception as e:
        logger.error(f"Direct LLM error: {e}", exc_info=True)
        return {
            "response": f"I encountered an error: {str(e)}",
            "tools_used": [],
            "error": str(e),
        }


async def _run_agent_stream_mock(
    message: str,
    chat_history: list[dict],
    require_approval: bool = False,
) -> AsyncGenerator[dict, None]:
    """Simulates the ReAct agent for streaming in Demo/Mock Mode with thought blocks and tool approvals."""
    try:
        msg_lower = message.lower()
        pdf_meta = get_pdf_metadata()
        
        tools_used = []
        tool_outputs = []
        
        # 1. Math check
        is_math = any(kw in msg_lower for kw in ["convert", "interest", "calculate", "math", "sin", "cos", "tan", "log", "sqrt", "+", "-", "*", "/", "^", "pow"]) or any(c.isdigit() for c in msg_lower)
        # 2. PDF check
        is_pdf = bool(pdf_meta and any(kw in msg_lower for kw in ["pdf", "document", "file", "uploaded", "summary", "summarize", "page", "text"]))
        # 3. Search check
        is_search = any(kw in msg_lower for kw in ["search", "weather", "news", "headlines", "price", "stock", "market", "who is", "latest", "current"])

        # Stream the mock thought process first
        thought_content = f"<thought>\nAnalyzing query: \"{message}\"\n"
        if is_pdf:
            thought_content += f"Detected reference to PDF document: '{pdf_meta.get('filename')}'\nDecided to run the pdf_qa_tool to fetch relevant text segments.\n"
        elif is_math:
            thought_content += "Detected mathematical expression. Will route to calculator_tool for AST-based computation.\n"
        elif is_search:
            thought_content += "Detected informational query requiring real-time updates. Routing to web_search.\n"
        else:
            thought_content += "General conversational text. No external tools required, answering using base capabilities.\n"
        thought_content += "</thought>\n\n"
        
        for i in range(0, len(thought_content), 5):
            yield _sse_event({
                "type": "token",
                "content": thought_content[i:i+5],
            })
            await asyncio.sleep(0.015)

        # Intercept with a mock approval if required
        if require_approval and (is_pdf or is_math or is_search):
            tool_name = "pdf_qa_tool" if is_pdf else ("calculator_tool" if is_math else "web_search")
            emoji = "📄" if is_pdf else ("🧮" if is_math else "🔍")
            approval_id = f"mock-approve-{uuid.uuid4().hex[:8]}"
            event = asyncio.Event()
            
            active_approvals[approval_id] = {
                "event": event,
                "approved": None,
                "tool": tool_name,
                "input": message,
            }
            
            yield _sse_event({
                "type": "tool_approval_request",
                "approval_id": approval_id,
                "tool": tool_name,
                "emoji": emoji,
                "label": f"Approval requested for {tool_name}",
                "input": message,
            })
            
            # Wait for user decision
            await event.wait()
            decision = active_approvals[approval_id]["approved"]
            active_approvals.pop(approval_id, None)
            
            if not decision:
                yield _sse_event({
                    "type": "token",
                    "content": "Tool execution was rejected by the user. I cannot proceed with the requested operation without tool access.",
                })
                yield _sse_event({"type": "done"})
                return

        # Execute Mock tool
        if is_pdf:
            tools_used.append("pdf_qa_tool")
            yield _sse_event({
                "type": "tool_start",
                "tool": "pdf_qa_tool",
                "emoji": "📄",
                "label": "Reading PDF...",
            })
            await asyncio.sleep(1.2)
            from tools.pdf_tool import create_pdf_tool
            pdf_qa = create_pdf_tool()
            tool_res = pdf_qa.invoke(message)
            tool_outputs.append(("pdf_qa_tool", tool_res))
            yield _sse_event({
                "type": "tool_end",
                "tool": "pdf_qa_tool",
                "output": str(tool_res),
            })
        elif is_math:
            tools_used.append("calculator_tool")
            yield _sse_event({
                "type": "tool_start",
                "tool": "calculator_tool",
                "emoji": "🧮",
                "label": "Calculating...",
            })
            await asyncio.sleep(0.8)
            from tools.calculator import calculator_tool as calc
            tool_res = calc.invoke(message)
            tool_outputs.append(("calculator_tool", tool_res))
            yield _sse_event({
                "type": "tool_end",
                "tool": "calculator_tool",
                "output": str(tool_res),
            })
        elif is_search:
            tools_used.append("web_search")
            yield _sse_event({
                "type": "tool_start",
                "tool": "web_search",
                "emoji": "🔍",
                "label": "Searching the web...",
            })
            await asyncio.sleep(1.0)
            from tools.search import web_search_tool as search
            tool_res = search.invoke(message)
            tool_outputs.append(("web_search", tool_res))
            yield _sse_event({
                "type": "tool_end",
                "tool": "web_search",
                "output": str(tool_res),
            })

        response_lines = []
        if tool_outputs:
            tool_name, tool_res = tool_outputs[0]
            if tool_name == "calculator_tool":
                response_lines.append(f"Using the AST Calculator tool, I computed the following result:\n\n{tool_res}")
            elif tool_name == "pdf_qa_tool":
                filename = pdf_meta.get("filename", "document")
                response_lines.append(f"I queried the document **{filename}** for your request.\n\nHere are the retrieved relevant excerpts:\n\n{tool_res}\n\nBased on these excerpts, I hope this helps clarify your question!")
            elif tool_name == "web_search":
                response_lines.append(f"Here are the web search results for your query:\n\n{tool_res}")
        else:
            if "hello" in msg_lower or "hi" in msg_lower:
                response_lines.append("Hello! I am your Multi-Tool AI Assistant running in **Demo Mode**.\n\nHow can I help you today?")
            elif "who are you" in msg_lower or "what can you do" in msg_lower:
                response_lines.append(
                    "I am a Multi-Tool AI Assistant.\n\n"
                    "In my full configuration, I orchestrate a LangChain ReAct agent calling:\n"
                    "- 🧮 **AST Calculator** for safe math\n"
                    "- 🔍 **Tavily Search** for real-time web lookup\n"
                    "- 📄 **ChromaDB Vector Store** for PDF Q&A\n\n"
                    "Since you are in Demo Mode, I am running a mock routing logic, but the actual Calculator, search fallbacks, and PDF ingestion (using FakeEmbeddings) are fully active!"
                )
            else:
                response_lines.append(
                    f"Thank you for your message: \"{message}\"\n\n"
                    "This is a general response generated in Demo Mode. If you upload a PDF, ask a math question, or query current events (like weather or news), I will automatically demonstrate the appropriate tool execution flow!"
                )

        response_lines.append("\n\n---\n\n> 💡 **Demo Mode Notice**\n> The application is currently running in local Demo/Mock Fallback Mode. To unlock full AI-powered ReAct agent flows, configure your actual Groq, Google Gemini, and Tavily API keys in `backend/.env`.")

        final_response = "\n".join(response_lines)
        
        chunk_size = 5
        for i in range(0, len(final_response), chunk_size):
            chunk = final_response[i:i+chunk_size]
            yield _sse_event({
                "type": "token",
                "content": chunk,
            })
            await asyncio.sleep(0.015)
            
        yield _sse_event({"type": "done"})

    except Exception as e:
        logger.error(f"Mock agent error: {e}", exc_info=True)
        yield _sse_event({
            "type": "error",
            "error": str(e),
        })


def _run_agent_sync_mock(
    message: str,
    chat_history: list[dict],
) -> dict:
    """Simulates the ReAct agent synchronously for non-streaming endpoints in Demo/Mock Mode."""
    msg_lower = message.lower()
    pdf_meta = get_pdf_metadata()
    
    tools_used = []
    tool_outputs = []
    
    is_math = any(kw in msg_lower for kw in ["convert", "interest", "calculate", "math", "sin", "cos", "tan", "log", "sqrt", "+", "-", "*", "/", "^", "pow"]) or any(c.isdigit() for c in msg_lower)
    is_pdf = bool(pdf_meta and any(kw in msg_lower for kw in ["pdf", "document", "file", "uploaded", "summary", "summarize", "page", "text"]))
    is_search = any(kw in msg_lower for kw in ["search", "weather", "news", "headlines", "price", "stock", "market", "who is", "latest", "current"])

    if is_pdf:
        tools_used.append("pdf_qa_tool")
        from tools.pdf_tool import create_pdf_tool
        pdf_qa = create_pdf_tool()
        tool_res = pdf_qa.invoke(message)
        tool_outputs.append(("pdf_qa_tool", tool_res))
    elif is_math:
        tools_used.append("calculator_tool")
        from tools.calculator import calculator_tool as calc
        tool_res = calc.invoke(message)
        tool_outputs.append(("calculator_tool", tool_res))
    elif is_search:
        tools_used.append("web_search")
        from tools.search import web_search_tool as search
        tool_res = search.invoke(message)
        tool_outputs.append(("web_search", tool_res))

    response_lines = []
    if tool_outputs:
        tool_name, tool_res = tool_outputs[0]
        if tool_name == "calculator_tool":
            response_lines.append(f"Using the AST Calculator tool, I computed the following result:\n\n{tool_res}")
        elif tool_name == "pdf_qa_tool":
            filename = pdf_meta.get("filename", "document")
            response_lines.append(f"I queried the document **{filename}** for your request.\n\nHere are the retrieved relevant excerpts:\n\n{tool_res}\n\nBased on these excerpts, I hope this helps clarify your question!")
        elif tool_name == "web_search":
            response_lines.append(f"Here are the web search results for your query:\n\n{tool_res}")
    else:
        if "hello" in msg_lower or "hi" in msg_lower:
            response_lines.append("Hello! I am your Multi-Tool AI Assistant running in **Demo Mode**.\n\nHow can I help you today?")
        elif "who are you" in msg_lower or "what can you do" in msg_lower:
            response_lines.append(
                "I am a Multi-Tool AI Assistant.\n\n"
                "In my full configuration, I orchestrate a LangChain ReAct agent calling:\n"
                "- 🧮 **AST Calculator** for safe math\n"
                "- 🔍 **Tavily Search** for real-time web lookup\n"
                "- 📄 **ChromaDB Vector Store** for PDF Q&A\n\n"
                "Since you are in Demo Mode, I am running a mock routing logic, but the actual Calculator, search fallbacks, and PDF ingestion (using FakeEmbeddings) are fully active!"
            )
        else:
            response_lines.append(
                f"Thank you for your message: \"{message}\"\n\n"
                "This is a general response generated in Demo Mode. If you upload a PDF, ask a math question, or query current events (like weather or news), I will automatically demonstrate the appropriate tool execution flow!"
            )

    response_lines.append("\n\n---\n\n> 💡 **Demo Mode Notice**\n> The application is currently running in local Demo/Mock Fallback Mode. To unlock full AI-powered ReAct agent flows, configure your actual Groq, Google Gemini, and Tavily API keys in `backend/.env`.")

    final_response = "\n".join(response_lines)
    return {
        "response": final_response,
        "tools_used": tools_used,
        "intermediate_steps": len(tools_used),
    }
