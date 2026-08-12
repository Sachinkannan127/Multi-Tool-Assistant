from langchain.tools import tool
from config import settings
import httpx

@tool
def slack_tool(channel: str, message: str) -> str:
    """Send a message to a Slack channel using the provided Slack Bot Token.
    
    Args:
        channel (str): The ID of the channel to post to (e.g. "C1234567890").
        message (str): The text message to send.
    """
    token = settings.slack_token
    if not token:
        return "Error: No Slack token configured. Please tell the user to add it in the Integrations tab in Settings."
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "channel": channel,
        "text": message
    }
    
    try:
        with httpx.Client() as client:
            res = client.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                if data.get("ok"):
                    return f"Message successfully sent to {channel}."
                return f"Slack API error: {data.get('error')}"
            return f"HTTP error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"
