import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion(data_path, index_root, index_name):
    print(f"--- Starting Ingestion from {data_path} ---")

    # Load Documents
    pdf_loader = DirectoryLoader(data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(data_path, glob="./*.txt", loader_cls=TextLoader)
    docs = pdf_loader.load() + txt_loader.load()
    
    if not docs:
        print("No documents found to process.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Initialize Azure Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Create Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to the nested path: index/faiss_index
    save_path = os.path.join(index_root, index_name)
    if not os.path.exists(index_root):
        os.makedirs(index_root)
        
    vectorstore.save_local(save_path)
    print(f"Success! FAISS index saved to: {save_path}")

if __name__ == "__main__":
    run_ingestion("data", "index", "faiss_index")