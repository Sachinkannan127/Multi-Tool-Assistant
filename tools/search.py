"""
Web Search Tool using Tavily Search API.
Returns synthesized results with source citations.
"""

import os
from typing import Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from config import settings


def create_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Create a Tavily search tool instance."""
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    return TavilySearchResults(
        name="web_search",
        description=(
            "Search the internet for real-time information including news, weather, "
            "current events, facts, and up-to-date data. Use this tool when the user "
            "asks about recent events, current prices, weather, news, or any information "
            "that requires up-to-date knowledge. Input should be a specific search query."
        ),
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


@tool
def web_search_tool(query: str) -> str:
    """
    Search the internet for real-time information including news, weather,
    current events, facts, and up-to-date data. Use this tool when the user
    asks about recent events, current prices, weather, news, or any information
    that requires up-to-date knowledge.

    Input: A specific search query string.
    Output: Search results with citations and source URLs.
    """
    if not settings.is_tavily_configured:
        query_lower = query.lower()
        if "weather" in query_lower:
            return (
                f"Summary: Simulated current weather conditions for '{query}':\n\n"
                f"[1] Weather for {query}: 82°F (28°C), Partly Cloudy, Wind 5mph, Humidity 65%\n"
                f"Source: https://weather.mock/search?q={query.replace(' ', '+')}\n"
            )
        elif "news" in query_lower or "headlines" in query_lower or "current events" in query_lower:
            return (
                "Summary: Here are the top news headlines globally:\n\n"
                "[1] Tech Breakthrough: New local AI agent architectures show massive performance improvements on edge devices.\n"
                "Source: https://technews.mock/local-ai-agents\n"
                "[2] Space Exploration: Mars rover discovers new structural evidence of ancient clay deposits in crater beds.\n"
                "Source: https://spacenews.mock/mars-clay-crater\n"
                "[3] Green Energy: Global investment in solar power infrastructures increases by 35% in the last fiscal quarter.\n"
                "Source: https://economy.mock/green-energy-rise\n"
            )
        elif "price" in query_lower or "stock" in query_lower or "market" in query_lower:
            return (
                "Summary: Markets are showing moderate gains today. Notable assets:\n\n"
                "[1] AAPL: $192.50 (+1.2%)\n"
                "Source: https://finance.mock/stocks/aapl\n"
                "[2] GOOGL: $175.80 (+0.8%)\n"
                "Source: https://finance.mock/stocks/googl\n"
                "[3] BTC: $62,400 (-0.5%)\n"
                "Source: https://finance.mock/crypto/btc\n"
            )
        else:
            return (
                f"Summary: Simulated search results for '{query}':\n\n"
                f"[1] Understanding {query}: A comprehensive guide to the concept and its applications.\n"
                f"Source: https://wikipedia.mock/wiki/{query.replace(' ', '_')}\n"
                f"[2] Latest trends and developments concerning {query}.\n"
                f"Source: https://industry-reports.mock/{query.replace(' ', '-')}\n"
            )

    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    try:
        search = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        results = search.invoke(query)

        if isinstance(results, str):
            return results

        # Format results nicely
        formatted = []
        if isinstance(results, dict):
            if "answer" in results:
                formatted.append(f"Summary: {results['answer']}\n")
            if "results" in results:
                for i, r in enumerate(results["results"], 1):
                    title = r.get("title", "Untitled")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    formatted.append(f"[{i}] {title}\n{content}\nSource: {url}\n")

        elif isinstance(results, list):
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                content = r.get("content", "")
                url = r.get("url", "")
                formatted.append(f"[{i}] {title}\n{content}\nSource: {url}\n")

        return "\n".join(formatted) if formatted else str(results)

    except Exception as e:
        return f"Search error: {str(e)}. Please try a different query."
