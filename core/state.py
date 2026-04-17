from typing import TypedDict, Annotated, Any
import operator

class AgentState(TypedDict):
    question: str
    session_id: str
    uploaded_files: list
    file_types: list
    route: str
    retry_count: int
    documents: list
    generation: str
    reflection: str
    dataframe_summaries: str
    figures: list  # Holds our generated Plotly chart objects