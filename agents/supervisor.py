from langchain_core.prompts import ChatPromptTemplate
from core.state import AgentState
from core.llm import get_llm

def supervisor_node(state: AgentState):
    """Analyzes the user's query and routes to the correct specialized agent."""
    print("--- [Node: Supervisor] Routing Query ---")
    llm = get_llm(temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent routing supervisor. 
    The user has uploaded the following file types in this session: {file_types}

    Analyze the user's input: "{question}"

    Decide the best route for this query:
    - Return "data_analysis" if the query requires calculating numbers, aggregating tabular data, or generating charts from Excel/CSV files.
    - Return "rag" if the query requires reading text, summarizing paragraphs, or extracting information from PDFs/Word documents.
    - Return "general" if it is a casual greeting or unrelated to the uploaded documents.

    Respond with ONLY the exact route name (rag, data_analysis, or general). No markdown, no explanations.
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "file_types": state.get("file_types", []),
        "question": state["question"]
    })
    
    route = response.content.strip().lower()
    
    # Fallback to general if the LLM hallucinated a route name
    if route not in ["rag", "data_analysis", "general"]:
        route = "general" 
        
    return {"route": route}