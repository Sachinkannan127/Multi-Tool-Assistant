"""Tools package for the Multi-Tool Personal Assistant."""

from tools.calculator import calculator_tool
from tools.search import create_search_tool
from tools.pdf_tool import create_pdf_tool, upload_and_index_pdf
from tools.memory import save_memory_tool, list_memories_tool, delete_memory_tool
from tools.scraper import web_scraper_tool
from tools.code_generator import code_generator_tool
from tools.youtube_tool import youtube_tool
from tools.instagram_tool import instagram_tool

__all__ = [
    "calculator_tool",
    "create_search_tool",
    "create_pdf_tool",
    "upload_and_index_pdf",
    "save_memory_tool",
    "list_memories_tool",
    "delete_memory_tool",
    "web_scraper_tool",
    "code_generator_tool",
    "youtube_tool",
    "instagram_tool",
]
