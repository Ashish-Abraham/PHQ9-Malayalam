
from langchain_core.messages import BaseMessage

def get_message_text(message: BaseMessage | str | list) -> str:
    """
    Safely extracts text content from a LangChain message, handling string, list, or direct string input.
    Useful for handling multimodal inputs from Gradio which come as lists.
    """
    if isinstance(message, BaseMessage):
        content = message.content
    else:
        content = message

    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return " ".join(text_parts)
    
    return str(content)
