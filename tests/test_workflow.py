import pytest
from core.session_store import SessionStore
from core.state import AgentState

def test_session_store_persistence():
    """Verify that data saved for one session doesn't leak into another."""
    store = SessionStore()
    store.set_dataframe("session_1", "data_1")
    store.set_dataframe("session_2", "data_2")
    
    assert store.get_dataframe("session_1") == "data_1"
    assert store.get_dataframe("session_2") == "data_2"

def test_agent_state_initialization():
    """Ensure the AgentState TypedDict contains all required keys."""
    state: AgentState = {
        "question": "Hello",
        "session_id": "test_id",
        "uploaded_files": [],
        "file_types": [],
        "route": "",
        "retry_count": 0,
        "documents": [],
        "generation": "",
        "reflection": "",
        "dataframe_summaries": ""
    }
    assert state["question"] == "Hello"
    assert state["retry_count"] == 0

def test_supervisor_logic_mock():
    """
    Test if the node properly updates the state dictionary when files are present.
    """
    from agents.supervisor import supervisor_node
    from unittest.mock import patch
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import AIMessage

    # FIX: MagicMock breaks LangChain's "|" operator. 
    # We must use a proper LangChain Runnable object to fake the LLM.
    fake_llm = RunnableLambda(lambda prompt: AIMessage(content="data_analysis"))

    # Mocking the get_llm call to return our fake LLM
    with patch('agents.supervisor.get_llm', return_value=fake_llm):
        
        initial_state = {
            "question": "Analyze my excel", 
            "file_types": [".xlsx"],
            "uploaded_files": ["dummy_data.xlsx"]
        }
        
        result = supervisor_node(initial_state)
        
        assert result["route"] == "data_analysis"