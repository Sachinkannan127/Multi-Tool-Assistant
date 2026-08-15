"""
Wikipedia Tool — search and retrieve information from Wikipedia.
"""

from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper

@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia and retrieve article summaries or content.
    
    Use this tool when:
    - The user asks for factual information about a person, place, event, or concept.
    - The user specifically mentions "Wikipedia" or asks to search Wikipedia.
    - You need detailed, encyclopedic background information on a topic.
    
    Input: A search query or topic (e.g., "Python programming language", "Albert Einstein")
    Output: A summary of the Wikipedia article(s) related to the query.
    """
    try:
        wikipedia = WikipediaAPIWrapper()
        return wikipedia.run(query)
    except Exception as e:
        return f"❌ Wikipedia search failed: {str(e)}"
