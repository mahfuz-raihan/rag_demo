from typing import Dict, Any

class SessionStore:
    """
    An ephemeral, in-memory store to hold un-serializable objects (like FAISS indices 
    and Pandas DataFrames) securely mapped by session_id. 
    This ensures data isolation per user and clears when the server restarts or session ends.
    """
    def __init__(self):
        self.vector_stores: Dict[str, Any] = {}
        self.dataframes: Dict[str, Any] = {}

    def get_vector_store(self, session_id: str):
        return self.vector_stores.get(session_id)

    def set_vector_store(self, session_id: str, vector_store: Any):
        self.vector_stores[session_id] = vector_store

    def get_dataframe(self, session_id: str):
        return self.dataframes.get(session_id)

    def set_dataframe(self, session_id: str, df: Any):
        self.dataframes[session_id] = df

# Global instance to be imported across agent nodes
session_store = SessionStore()