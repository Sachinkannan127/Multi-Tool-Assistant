from langchain_core.tools import tool
import memory_store

@tool
def save_memory_tool(content: str) -> str:
    """
    Save a persistent memory or fact about the user (e.g. user preferences,
    background, facts, or instructions that should be remembered in future chats).
    
    Input: The specific fact or description to remember (e.g. "User's name is Alex" or "User prefers python code instead of javascript").
    Output: A success message confirming the memory has been saved.
    """
    new_mem = memory_store.add_memory(content)
    return f"Success: Memory saved. ID: {new_mem['id']}"

@tool
def list_memories_tool() -> str:
    """
    Retrieve all persistent memories, facts, and notes saved about the user.
    Use this tool at the start of a conversation if you need to recall who the user is or what their preferences are.
    
    Output: A list of saved memories with their IDs and timestamps.
    """
    memories = memory_store.load_memories()
    if not memories:
        return "No memories found in the store."
    
    lines = []
    for m in memories:
        lines.append(f"- [{m['id']}] (saved at {m['timestamp']}): {m['content']}")
    return "\n".join(lines)

@tool
def delete_memory_tool(memory_id: str) -> str:
    """
    Delete a persistent memory or fact by its ID when it is no longer valid,
    incorrect, or has been updated by new information.
    
    Input: The ID of the memory to delete (e.g., 'mem_1a2b3c4d').
    Output: A message confirming deletion status.
    """
    success = memory_store.delete_memory(memory_id)
    if success:
        return f"Success: Memory with ID {memory_id} deleted."
    return f"Error: Memory with ID {memory_id} not found."
