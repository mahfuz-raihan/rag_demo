from typing import List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str          # my query
    generation: str        # ai generated response
    documents: List[Document] # chunks retrieved from FAISS
    reflection: str        # critique/score from the self-reflection step
    retry_count: int       # prevent infinite loops during reflection