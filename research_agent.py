import json
import logging
import asyncio
from typing import AsyncGenerator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel

from config import settings
from agent import get_llm, _sse_event, _is_rate_limit_error
from tools.search import web_search_tool
from tools.scraper import web_scraper_tool
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

def create_research_executor(temperature: float = 0.2) -> AgentExecutor:
    """Create a specialized agent executor for deep research."""
    llm = get_llm(temperature=temperature)  # Use the user's configured provider from settings
    
    tools = [web_search_tool, web_scraper_tool]

    sys_prompt = (
        "You are an autonomous Deep Research Agent. Your goal is to compile a massive, highly detailed, "
        "and accurate research report on the user's topic.\n\n"
        "INSTRUCTIONS:\n"
        "1. You MUST use the `web_search` tool to find the most relevant and up-to-date sources.\n"
        "2. You MUST use the `web_scraper` tool to read the full content of at least 2-3 highly relevant URLs you found.\n"
        "3. Synthesize the information from these multiple sources.\n"
        "4. Your final output MUST be a comprehensive Markdown report. Include sections, bullet points, and deep technical details if applicable.\n"
        "5. You MUST include inline citations (e.g. [1]) and a 'Sources' section at the end linking back to the URLs you scraped.\n\n"
        "Do not stop until you have gathered sufficient deep knowledge to write a definitive report."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,       # Allow many steps for deep research
        max_execution_time=300,  # Allow up to 5 minutes
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor

def create_fallback_research_executor() -> AgentExecutor:
    """Create a fallback agent executor using a smaller/faster model to bypass rate limits."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=settings.groq_api_key,
        temperature=0.2,
        streaming=True,
        max_tokens=4096,
        max_retries=0,
    )
    tools = [web_search_tool, web_scraper_tool]
    
    sys_prompt = (
        "You are an autonomous Deep Research Agent. Your goal is to compile a detailed "
        "and accurate research report on the user's topic.\n\n"
        "INSTRUCTIONS:\n"
        "1. Use `web_search` to find sources.\n"
        "2. Use `web_scraper` to read 1-2 highly relevant URLs.\n"
        "3. Your final output MUST be a comprehensive Markdown report with citations (e.g. [1]).\n"
        "Do not stop until you have gathered sufficient knowledge."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=6, 
        max_execution_time=120, return_intermediate_steps=True, handle_parsing_errors=True,
    )


async def run_research_stream(topic: str, **kwargs) -> AsyncGenerator[dict, None]:
    """Run the deep research agent and stream back steps and markdown tokens."""
    try:
        if kwargs.get('is_fallback'):
            executor = create_fallback_research_executor()
        else:
            executor = create_research_executor()
        
        # Stream events and tokens in real-time
        async for event in executor.astream_events(
            {"input": f"Research Topic: {topic}"},
            version="v2"
        ):
            kind = event.get("event")
            name = event.get("name")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    yield _sse_event({
                        "type": "token",
                        "content": content,
                    })

            elif kind == "on_tool_start":
                tool_name = name
                tool_input = event["data"].get("input")
                
                # Format a nice readable message for the UI
                msg = f"Using {tool_name}..."
                if tool_name == "web_search":
                    q = tool_input.get("query", "") if isinstance(tool_input, dict) else tool_input
                    msg = f"Searching web for: {q}"
                elif tool_name == "web_scraper":
                    url = tool_input.get("url", "") if isinstance(tool_input, dict) else tool_input
                    msg = f"Reading article: {url}"

                yield _sse_event({
                    "type": "research_step",
                    "tool": tool_name,
                    "message": msg,
                    "input": tool_input,
                })

            elif kind == "on_tool_end":
                tool_name = name
                yield _sse_event({
                    "type": "tool_end",
                    "tool": tool_name,
                })

        yield _sse_event({"type": "done"})

    except Exception as e:
        logger.error(f"Research agent error: {e}")
        
        # If rate limited, attempt to fallback to the lighter 8b model
        if _is_rate_limit_error(e) and not kwargs.get('is_fallback'):
            yield _sse_event({
                "type": "research_step",
                "tool": "system",
                "message": "Rate limit hit! Falling back to backup model...",
                "input": "",
            })
            try:
                # Recursively call with fallback flag
                async for event in run_research_stream(topic, is_fallback=True):
                    yield event
                return
            except Exception as e2:
                logger.error(f"Fallback research agent error: {e2}")
                yield _sse_event({
                    "type": "error",
                    "error": "Rate limit exceeded on all models. Please wait a few minutes and try again.",
                })
                return

        # Format user-friendly error
        err_msg = str(e)
        if _is_rate_limit_error(e):
            err_msg = "Rate limit exceeded on all models. Please wait a few minutes and try again."
            
        yield _sse_event({
            "type": "error",
            "error": err_msg,
        })
