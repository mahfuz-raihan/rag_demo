from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from core.state import AgentState
from core.llm import get_llm
from core.session_store import session_store

def data_analyst_node(state: AgentState):
    """Use Pandas agent to analyze tabular data dynamically."""
    print("--- [Node: Data Analyst] Analyzing tabular data ---")
    session_id = state.get("session_id")
    df = session_store.get_dataframe(session_id)
    
    if df is None:
        return {"generation": "No tabular data (Excel/CSV) was found in your uploaded files to analyze."}
        
    llm = get_llm(temperature=0)
    
    # Create the agent
    # allow_dangerous_code=True is required because the agent writes and executes pandas code under the hood.
    # Since we are deploying this in a Docker container on Render, this is isolated and safe.
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True, 
        agent_type="openai-tools"
    )
    
    try:
        response = agent.invoke({"input": state["question"]})
        generation = response.get("output", "Could not generate analysis.")
    except Exception as e:
        generation = f"An error occurred while analyzing the data: {str(e)}"
        
    return {"generation": generation}