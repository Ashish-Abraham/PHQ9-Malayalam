import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import create_graph

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

def main():
    print("Initializing PHQ-9 Chatbot...")
    graph = create_graph()
    
    print("Chatbot ready. Type 'quit' to exit.")
    print("Bot: Hello! I'm here to listen. How are you feeling today?")
    
    # Initial state
    state = {
        "messages": [],
        "phase": "rapport",
        "phq9_responses": {},
        "current_question_index": 0,
        "patient_info": "",
        "financial_distress": "",
        "study_pressure": "",
        "permission_granted": False
    }
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph
        # We need to handle the recursion limit or just run one step?
        # graph.invoke(state) runs until end or interruption.
        # But we want to stop after the bot responds to get user input.
        # LangGraph runs until it hits a node that requires input? No, it runs until END or a breakpoint.
        # But our graph has cycles (rapport -> rapport).
        # So we need to run it such that it stops after generating a response.
        # In our nodes, we return a response.
        # If we just invoke, it might loop infinitely if the router sends it back to the same node 
        # AND the node doesn't wait for user input.
        # BUT LangGraph nodes are functions. They run once.
        # If the edge goes back to 'rapport', it will run 'rapport' again immediately?
        # YES.
        # We need to use `interrupt_before` or `interrupt_after` or just structure it differently.
        # OR, we treat the graph execution as "process one turn".
        # But LangGraph is designed to run until completion.
        # For chatbots, we usually use `checkpointer` or just run until it stops.
        # But here, 'rapport' -> 'rapport' is an infinite loop if we don't break.
        # WE NEED TO BREAK to get user input.
        # The standard pattern is: Node generates response -> END (of this turn).
        # But we want to maintain state.
        
        # Actually, for a chatbot, the "User" is usually not a node in the graph (unless using HumanInTheLoop).
        # The graph processes the input and produces output.
        # So the edge should be: Rapport -> END.
        # And then next time we call invoke, we start from where?
        # LangGraph with Checkpointer allows resuming.
        # WITHOUT Checkpointer, we pass the full state every time.
        # So, if we want to stop, the edge should go to END.
        # But we have conditional edges.
        
        # Let's modify the graph to return to END after each bot response, 
        # but we need to know which node to start from next time?
        # No, we always pass the state. The state has 'phase'.
        # We can use the 'phase' to route to the correct node at the START.
        # So we need a 'router' entry point.
        
        result = graph.invoke(state)
        state = result # Update state
        
        # Print the last message from bot
        if state["messages"] and state["messages"][-1].type == "ai":
            print(f"Bot: {state['messages'][-1].content}")
            
        if state.get("phase") == "end":
            print("Chatbot session ended.")
            break
            
if __name__ == "__main__":
    main()
