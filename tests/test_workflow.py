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
    from unittest.mock import MagicMock, patch

    # Mocking the get_llm call so we don't spend money during testing
    with patch('agents.supervisor.get_llm') as mock_llm:
        
        # 1. We mock the LLM's response
        mock_response = MagicMock()
        
        # Note: If your supervisor uses structured outputs (Pydantic), 
        # it accesses object properties. If it uses raw text, it uses .content.
        # We handle both just to be safe!
        mock_response.content = "data_analysis"
        mock_response.route = "data_analysis" 
        
        mock_llm.return_value.invoke.return_value = mock_response
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_response

        # 2. FIX: We add "uploaded_files" to trick the supervisor into knowing a file exists
        initial_state = {
            "question": "Analyze my excel", 
            "file_types": [".xlsx"],
            "uploaded_files": ["dummy_data.xlsx"] # <--- THE FIX
        }
        
        result = supervisor_node(initial_state)
        
        assert result["route"] == "data_analysis"