"""
Instagram Tool — extracts data from Instagram posts or profiles using Apify.

Uses apify-client and the apify/instagram-scraper actor.
"""

import os
import json
import logging
from typing import Optional
from langchain_core.tools import tool
from config import settings

logger = logging.getLogger(__name__)

def _fetch_instagram_data_sync(url: str) -> dict:
    """
    Synchronously scrape an Instagram URL using Apify.
    """
    # Check if API token is available
    apify_token = settings.apify_token or os.getenv("APIFY_TOKEN")
    if not apify_token:
        return {"error": "APIFY_TOKEN is missing. Please configure it in settings."}

    try:
        from apify_client import ApifyClient
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Please install apify-client."}

    try:
        # Initialize the ApifyClient with API token
        client = ApifyClient(apify_token)

        # We use the popular 'apify/instagram-scraper' actor
        run_input = {
            "directUrls": [url],
            "resultsType": "details",
            "resultsLimit": 5, # Just in case it's a profile, keep it small to be fast
        }
        
        # Run the Actor and wait for it to finish
        # apify/instagram-scraper is a common actor ID
        run = client.actor("apify/instagram-scraper").call(run_input=run_input)

        if not run or "defaultDatasetId" not in run:
            return {"error": "Failed to retrieve data from Apify run."}

        # Fetch results from the dataset
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if not items:
            return {"error": "No data found for this Instagram URL."}
            
        # Format the scraped items into a readable text
        extracted_texts = []
        for item in items:
            text_parts = []
            if "ownerUsername" in item:
                text_parts.append(f"**Username:** @{item['ownerUsername']}")
            if "ownerFullName" in item:
                text_parts.append(f"**Full Name:** {item['ownerFullName']}")
            if "caption" in item:
                text_parts.append(f"**Caption:** {item['caption']}")
            if "likesCount" in item:
                text_parts.append(f"**Likes:** {item['likesCount']}")
            if "commentsCount" in item:
                text_parts.append(f"**Comments:** {item['commentsCount']}")
            if "timestamp" in item:
                text_parts.append(f"**Posted at:** {item['timestamp']}")
            if "type" in item:
                text_parts.append(f"**Post Type:** {item['type']}")
                
            extracted_texts.append("\n".join(text_parts))
            
        text = "\n\n---\n\n".join(extracted_texts)
            
        return {
            "text": text,
            "error": None
        }
    except Exception as e:
        logger.error(f"Instagram Apify error for {url}: {e}")
        return {"error": f"Failed to fetch Instagram data: {str(e)}"}


@tool
def instagram_tool(url: str) -> str:
    """
    Fetch data from an Instagram URL (post, reel, or profile) using Apify.
    
    Use this tool when:
    - The user provides an Instagram URL and asks to summarize, read, or extract information from it.
    - You need to know the caption, likes, comments, or metadata of an Instagram post.
    
    Input: A valid Instagram URL (e.g., https://www.instagram.com/p/...)
    Output: Extracted text including caption, likes, username, and post metadata.
    """
    result = _fetch_instagram_data_sync(url)
    
    if result.get("error"):
        return f"❌ Instagram Tool failed: {result['error']}"
        
    lines = []
    lines.append(f"# Instagram Data Extraction")
    lines.append(f"🔗 URL: {url}\n")
    lines.append(result["text"])
    
    return "\n".join(lines)
