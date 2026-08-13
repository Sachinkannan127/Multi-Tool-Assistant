"""
YouTube Tool — extracts transcripts from YouTube videos.

Uses youtube-transcript-api.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    try:
        parsed = urlparse(url)
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed.path[1:]
        if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
            if parsed.path == '/watch':
                return parse_qs(parsed.query)['v'][0]
            if parsed.path.startswith('/embed/'):
                return parsed.path.split('/')[2]
            if parsed.path.startswith('/v/'):
                return parsed.path.split('/')[2]
            if parsed.path.startswith('/shorts/'):
                return parsed.path.split('/')[2]
    except Exception as e:
        logger.error(f"Failed to parse YouTube URL {url}: {e}")
    return ""

def _fetch_transcript_sync(url: str) -> dict:
    """
    Synchronously fetch the transcript for a YouTube video.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import TextFormatter
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Please install youtube-transcript-api."}

    video_id = _extract_video_id(url)
    if not video_id:
        return {"error": f"Could not extract a valid YouTube video ID from URL: {url}"}

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript)
        
        # Truncate if extremely long (keep around 15k characters for LLM limits)
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...transcript truncated at {max_chars} characters]"
            
        return {
            "video_id": video_id,
            "text": text,
            "error": None
        }
    except Exception as e:
        logger.error(f"YouTube transcript error for {video_id}: {e}")
        return {"error": f"Failed to fetch transcript: {str(e)}\n(Note: The video might not have captions enabled.)"}

@tool
def youtube_tool(url: str) -> str:
    """
    Fetch the transcript of a YouTube video given its URL.
    
    Use this tool when:
    - The user provides a YouTube URL and asks to summarize, analyze, or discuss the video content.
    - The user wants to know what was said in a specific YouTube video.
    
    Input: A valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)
    Output: The textual transcript of the video.
    """
    result = _fetch_transcript_sync(url)
    
    if result.get("error"):
        return f"❌ YouTube Tool failed: {result['error']}"
        
    lines = []
    lines.append(f"# YouTube Video Transcript (ID: {result['video_id']})")
    lines.append(f"🔗 URL: {url}\n")
    lines.append(result["text"])
    
    return "\n".join(lines)
