import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState

load_dotenv()

# --- Configuration ---
# Update this to match your folder structure
INDEX_DIRECTORY = "index"
INDEX_NAME = "faiss_index"
FULL_INDEX_PATH = os.path.join(INDEX_DIRECTORY, INDEX_NAME)

# Initialize Azure OpenAI Components
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# Load Vector Store from the new path
if os.path.exists(FULL_INDEX_PATH):
    print(f"--- Loading index from {FULL_INDEX_PATH} ---")
    vectorstore = FAISS.load_local(
        FULL_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
else:
    print(f"--- WARNING: Index not found at {FULL_INDEX_PATH} ---")
    retriever = None

def retrieve_node(state: AgentState):
    """Fetch relevant documents from FAISS."""
    print("--- [Node: Retrieve] Searching knowledge base ---")
    if retriever is None:
        return {"documents": []}
    
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate_node(state: AgentState):
    """Generate answer based on retrieved context."""
    print("--- [Node: Generate] Synthesizing answer ---")
    question = state["question"]
    documents = state.get("documents", [])
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert domain assistant. Use the following context to answer the question.
    Look carefully at the headers and footers of the provided chunks for relevant information.
    If the context doesn't contain the answer, say you don't have enough information to answer.
    
    Context: {context}
    Question: {question}
    Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response.content}

def reflect_node(state: AgentState):
    """Critique the generated answer."""
    print("--- [Node: Reflect] Critiquing answer ---")
    question = state["question"]
    generation = state.get("generation", "")
    current_retries = state.get("retry_count", 0)
    
    prompt = ChatPromptTemplate.from_template("""
    Evaluate the answer below for accuracy and completeness based on the question.
    If the answer is sufficient, output only 'accurate'.
    If it is vague or wrong, output only 'needs_revision'.
    
    Question: {question}
    Answer: {generation}
    Result:
    """)
    
    critic_chain = prompt | llm
    reflection_res = critic_chain.invoke({"question": question, "generation": generation})
    status = "accurate" if "accurate" in reflection_res.content.lower() else "needs_revision"
    
    return {
        "reflection": status,
        "retry_count": current_retries + 1
    }