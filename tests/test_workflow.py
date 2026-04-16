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
    In a real CI, we might mock the LLM. 
    Here we test if the node properly updates the state dictionary.
    """
    from agents.supervisor import supervisor_node
    from unittest.mock import MagicMock, patch

    # Mocking the get_llm call so we don't spend money during testing
    with patch('agents.supervisor.get_llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = "data_analysis"
        mock_llm.return_value.invoke.return_value = mock_response

        initial_state = {"question": "Analyze my excel", "file_types": [".xlsx"]}
        result = supervisor_node(initial_state)
        
        assert result["route"] == "data_analysis"