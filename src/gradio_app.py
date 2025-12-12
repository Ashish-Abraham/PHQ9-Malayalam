import gradio as gr
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.graph import create_graph
from src.dashboard_app import create_dashboard

# Initialize graph
graph = create_graph()

def init_state():
    """Initialize the chat state."""
    return {
        "messages": [],
        "phase": "rapport",
        "phq9_responses": {},
        "current_question_index": 0,
        "patient_info": "",
        "financial_distress": "",
        "study_pressure": "",
        "permission_granted": False
    }

def chat_logic(message, history, state):
    """
    Core chat logic to be used by Gradio and tests.
    
    Args:
        message (str): The user's message.
        history (list): Chat history (unused by graph, but provided by Gradio).
        state (dict): The current conversation state.
        
    Returns:
        tuple: (response_message, updated_state)
    """
    if state is None:
        state = init_state()
        
    # Append user message to state
    state["messages"].append(HumanMessage(content=message))
    
    # Run the graph
    # The graph is designed to run until it hits a node that goes to END.
    # We invoke it with the current state.
    # Run the graph
    # The graph is designed to run until it hits a node that goes to END.
    # We invoke it with the current state.
    result = graph.invoke(state)
    
    # Update state with result
    state = result
    
    # Get the last AI message
    response = ""
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        response = state["messages"][-1].content
    elif state["messages"]:
        # Fallback if the last message isn't AIMessage (shouldn't happen with this graph)
        response = state["messages"][-1].content

    # --- Background Analysis for Dashboard ---
    import threading
    from src.shared_state import update_emotion, update_suicide_risk
    from src.utils.pipelines import detect_emotion, detect_suicidal_language
    
    def run_analysis(msg):
        try:
            # Detect Emotion
            emo = detect_emotion(msg)
            update_emotion(emo)
            
            # Detect Suicide Risk
            is_risk = detect_suicidal_language(msg)
            if is_risk:
                 update_suicide_risk({"alert": True, "text": msg})
        except Exception as e:
            print(f"Background analysis failed: {e}")

    # Fire and forget thread (if pipelines are NOT disabled)
    if os.environ.get("DISABLE_PIPELINES"):
        print("[System] Pipelines disabled by configuration. Background analysis skipped.")
    else:
        threading.Thread(target=run_analysis, args=(message,), daemon=True).start()
        
    return response, state

def gradio_chat(message, history, state):
    """Wrapper for Gradio chat interface."""
    response, new_state = chat_logic(message, history, state)
    return response, new_state

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# PHQ-9 Chatbot")
        
        state = gr.State(init_state())
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Type your message here...")
        clear = gr.Button("Clear")
        
        def user(user_message, history):
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history, current_state):
            # history is now a list of dicts. The last one is the user message.
            user_message = history[-1]["content"]
            # We don't pass history to chat_logic, but chat_logic signature expects it.
            # chat_logic doesn't use history, so safe to pass the new format.
            bot_message, new_state = gradio_chat(user_message, history[:-1], current_state)
            
            # Append bot response
            history.append({"role": "assistant", "content": bot_message})
            return history, new_state

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, state], [chatbot, state]
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    with demo.route("Dashboard", "/dashboard"):
        create_dashboard().render()
        
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
