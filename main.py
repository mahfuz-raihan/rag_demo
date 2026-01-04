import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from node import retrieve_node, generate_node, reflect_node

load_dotenv()

# 1. Define the Workflow
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflect_node)

# 3. Build the Graph Connections
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "reflect")

# 4. Conditional Logic for Self-Reflection
def decide_to_finish(state: AgentState):
    """Route back to retrieve if critic is unhappy, or end if accurate."""
    reflection = state.get("reflection", "")
    retry_count = state.get("retry_count", 0)
    
    # Check if we should stop
    if reflection == "accurate" or retry_count >= 3:
        return "end_node"
    else:
        print(f"--- [Logic] Rejected (Attempt {retry_count}). Searching again... ---")
        return "retry_node"

# FIXED: Explicit mapping of return values to node names/END
workflow.add_conditional_edges(
    "reflect",
    decide_to_finish,
    {
        "end_node": END,
        "retry_node": "retrieve"
    }
)

# 5. Compile the App
app = workflow.compile()

def run_chat():
    print("\n" + "="*50)
    print("AGENTIC DOMAIN QA SYSTEM")
    print("="*50)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            break
            
        # Initialize state
        current_state = {
            "question": user_input, 
            "retry_count": 0, 
            "documents": [], 
            "generation": "", 
            "reflection": ""
        }
        
        final_generation = ""
        
        # Stream the graph execution
        try:
            for output in app.stream(current_state):
                # The output is a dict where keys are node names
                for node_name, state_update in output.items():
                    if "generation" in state_update:
                        final_generation = state_update["generation"]
            
            print(f"\nAgent: {final_generation}")
        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")

if __name__ == "__main__":
    run_chat()