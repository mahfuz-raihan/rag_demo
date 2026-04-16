from core.state import AgentState
from core.llm import get_llm

def general_node(state: AgentState):
    """Handles casual conversation (like 'Hi', 'Hey') when no files are needed."""
    print("--- [Node: General] Conversational reply ---")
    
    # We use a slightly higher temperature for casual chat to be conversational
    llm = get_llm(temperature=0.7) 
    
    # Simply pass the user's question to the Azure LLM
    response = llm.invoke(state["question"])
    
    return {"generation": response.content}