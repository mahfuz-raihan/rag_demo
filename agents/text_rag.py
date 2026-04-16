from langchain_core.prompts import ChatPromptTemplate
from core.state import AgentState
from core.llm import get_llm
from core.session_store import session_store

def retrieve_node(state: AgentState):
    """Fetch relevant documents from the user's ephemeral FAISS index."""
    print("--- [Node: Retrieve] Searching knowledge base ---")
    session_id = state.get("session_id")
    vector_store = session_store.get_vector_store(session_id)
    
    if not vector_store:
        return {"documents": []}
        
    # K=5 retrieves top 5 relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    documents = retriever.invoke(state["question"])
    
    return {"documents": documents}

def generate_node(state: AgentState):
    """Generate an answer based on the retrieved context."""
    print("--- [Node: Generate] Synthesizing answer ---")
    llm = get_llm(temperature=0)
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"generation": "I couldn't find any relevant text documents uploaded to answer this."}
        
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert domain assistant. Use the following context from uploaded files to answer the question.
    If the context doesn't contain the answer, say you don't have enough information to answer.
    
    Context: {context}
    Question: {question}
    Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"generation": response.content}