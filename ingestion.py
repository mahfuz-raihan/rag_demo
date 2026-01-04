import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion(data_path: str, index_root: str, index_name: str):
    print(f"--- Starting Universal Ingestion from {data_path} ---")

    # 1. Define specific loaders for different file types
    # This mapping tells the DirectoryLoader which class to use for each extension
    loader_mapping = {
        "./*.pdf": PyPDFLoader,
        "./*.txt": TextLoader,
        "./*.docx": UnstructuredWordDocumentLoader,
        "./*.xlsx": UnstructuredExcelLoader,
    }

    all_docs = []
    
    # 2. Iterate through the mapping and load files
    for glob_pattern, loader_cls in loader_mapping.items():
        print(f"Loading files matching {glob_pattern}...")
        loader = DirectoryLoader(
            data_path, 
            glob=glob_pattern, 
            loader_cls=loader_cls,
            show_progress=True,
            use_multithreading=True
        )
        try:
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {glob_pattern}: {e}")

    if not all_docs:
        print("No documents found! Please ensure files are in the 'data' folder.")
        return

    print(f"Total documents loaded: {len(all_docs)}")

    # 3. Chunking
    # For Excel/Docs, keeping a decent chunk size is important for tabular data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # 4. Initialize Azure Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # 5. Create and Save Vector Store
    print("Generating embeddings and creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_path = os.path.join(index_root, index_name)
    if not os.path.exists(index_root):
        os.makedirs(index_root)
        
    vectorstore.save_local(save_path)
    print(f"Success! Universal index saved to: {save_path}")

if __name__ == "__main__":
    # Ensure you have 'data' and 'index' folders
    run_ingestion("data", "index", "faiss_index")