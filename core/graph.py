from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.supervisor import supervisor_node
from agents.text_rag import retrieve_node, generate_node
from agents.data_analyst import data_analyst_node
from agents.critic_transformer import reflect_node, transform_query_node
from agents.general import general_node

# 1. Define the workflow
workflow = StateGraph(AgentState)

# 2. Add all our agent nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("data_analyst", data_analyst_node)
workflow.add_node("critic", reflect_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("general", general_node)

# 3. Define the routing logic from the supervisor
def route_from_supervisor(state: AgentState):
    route = state.get("route", "general")
    if route == "data_analysis":
        return "data_analyst"
    elif route == "rag":
        return "retrieve"
    else:
        return "general"

# 4. Define the reflection/retry logic
def route_from_critic(state: AgentState):
    reflection = state.get("reflection", "")
    retry_count = state.get("retry_count", 0)
    
    if reflection == "accurate" or retry_count >= 3:
        return "end_node"
    else:
        print(f"--- [Logic] Rejected (Attempt {retry_count}). Transforming query... ---")
        return "transform_query"

# 5. Build the Graph Connections
workflow.set_entry_point("supervisor")

# Supervisor routes to either RAG, Data Analyst, or General Chat
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "retrieve": "retrieve",
        "data_analyst": "data_analyst", # Fixed: Now routes to the actual data_analyst node!
        "general": "general"
    }
)

# Text RAG standard flow
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critic")

# Data Analyst flow: Send output to critic after analyzing
workflow.add_edge("data_analyst", "critic")

# Critic evaluates and conditionally loops
workflow.add_conditional_edges(
    "critic",
    route_from_critic,
    {
        "end_node": END,
        "transform_query": "transform_query"
    }
)

workflow.add_edge("transform_query", "supervisor")
workflow.add_edge("general", END)

# Compile the graph
compiled_graph = workflow.compile()