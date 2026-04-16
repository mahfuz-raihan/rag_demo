from langchain_core.prompts import ChatPromptTemplate
from core.state import AgentState
from core.llm import get_llm

def reflect_node(state: AgentState):
    """Critique the generated answer to ensure quality."""
    print("--- [Node: Reflect] Critiquing answer ---")
    llm = get_llm(temperature=0)
    question = state["question"]
    generation = state.get("generation", "")
    current_retries = state.get("retry_count", 0)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a strict quality assurance evaluator. 
    Compare the user's original question with the generated answer.

    User Question: {question}
    Generated Answer: {generation}

    Evaluate the answer based on accuracy, completeness, and relevance.
    - If the answer sufficiently addresses the question, output ONLY 'accurate'.
    - If the answer is vague, says "I don't know", or fails to execute code properly, output ONLY 'needs_revision'.

    Do not provide any other text or reasoning.
    """)
    
    critic_chain = prompt | llm
    reflection_res = critic_chain.invoke({"question": question, "generation": generation})
    status = "accurate" if "accurate" in reflection_res.content.lower() else "needs_revision"
    
    return {
        "reflection": status,
        "retry_count": current_retries + 1
    }

def transform_query_node(state: AgentState):
    """Rewrite the query to fix the 'Insanity Loop'."""
    print("--- [Node: Transform Query] Rewriting query ---")
    llm = get_llm(temperature=0)
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert query optimizer. 
    The following user question failed to yield a good answer from our internal knowledge base or data tools: 

    Original Question: {question}

    Your task is to rephrase this question to make it better for semantic search or data extraction. 
    - Extract the core intent.
    - Use specific keywords.
    - Do NOT answer the question. Just output the optimized question.
    """)
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    optimized_query = response.content.strip()
    
    print(f"--- Optimized Query: {optimized_query} ---")
    return {"question": optimized_query}