from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.tools.rag import search_guidelines

TRIGGER_KEYWORDS = [
    "protocol", "guideline", "rule", "score", "severe", "mild", 
    "moderate", "cutoff", "interpretation", "policy", "faq"
]

def run_llm_with_rag(llm, messages):
    """
    Runs the LLM with the search_guidelines tool bound.
    Implements a strict fallback: if no tool call is made but keywords are present,
    it forces the tool execution.
    """
    # Bind tool
    llm_with_tools = llm.bind_tools([search_guidelines])
    
    # 1. Initial LLM Call
    response = llm_with_tools.invoke(messages)
    
    # 2. Check for Tool Call
    if response.tool_calls:
        # Execute tool calls
        tool_outputs = []
        for tool_call in response.tool_calls:
            # We only have one tool for now
            if tool_call["name"] == "search_guidelines":
                tool_output = search_guidelines.invoke(tool_call)
                tool_outputs.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"]
                    )
                )
        
        # Append tool outputs and get final response
        messages_with_tools = messages + [response] + tool_outputs
        final_response = llm_with_tools.invoke(messages_with_tools)
        return final_response
        
    # 3. Strict Fallback: No tool call, but maybe missed keywords?
    else:
        last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if last_user_msg:
            from src.utils.message_utils import get_message_text
            content = get_message_text(last_user_msg).lower()
            
            if any(keyword in content for keyword in TRIGGER_KEYWORDS):
                print(f"FALLBACK TRIGGERED: Keywords found in '{content}' but no tool call.")
                
                # Force search
                # We construct a fake tool call or just inject context.
                # Injecting context is safer/easier than forging a tool call chain if the model didn't start it.
                
                search_result = search_guidelines.invoke(last_user_msg.content)
                
                # Create a specialized system message with the context
                context_msg = SystemMessage(
                    content=f"IMPORTANT CONTEXT RETRIEVED (The user asked about guidelines and you missed it, so here it is):\n{search_result}\n\nUse this context to answer the user's previous question."
                )
                
                # Send back to LLM (without tools is fine, or with tools)
                messages_with_context = messages + [context_msg]
                fallback_response = llm.invoke(messages_with_context)
                return fallback_response

    # Normal response
    return response
