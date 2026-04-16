from typing import List, TypedDict, Any, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represents the state of our agentic workflow.
    """
    # User Input & Session
    question: str    # The user's current query
    session_id: str  # Unique ID for the user's current session
    
    # File Context (Dynamic Data)
    uploaded_files: List[Any]  # Raw file paths or file objects uploaded by user
    file_types: List[str]  # e.g., ['.pdf', '.xlsx', '.csv'] to help Supervisor
    
    # Routing & Flow Control
    route: str  # 'rag', 'data_analysis', or 'general'
    retry_count: int  # Prevents infinite reflection loops
    
    # RAG Context
    documents: List[Document]  # Chunks retrieved from ephemeral FAISS
    
    # Generation & Evaluation
    generation: str  # The AI's drafted response
    reflection: str  # 'accurate' or 'needs_revision' from Critic
    
    # Tabular Data Context 
    dataframe_summaries: str  # String representation of df.head() or schema