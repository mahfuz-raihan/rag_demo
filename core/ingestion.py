import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from core.llm import get_embeddings
from core.session_store import session_store

def process_attached_files(files, session_id: str):
    """
    Takes files attached via the Chainlit paperclip icon and processes them 
    into memory (FAISS for text, Pandas for tabular) tied to the user's session.
    """
    text_docs = []
    
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        
        # 1. Handle Tabular Data (Excel/CSV)
        if ext in ['.csv', '.xlsx', '.xls']:
            try:
                if ext == '.csv':
                    df = pd.read_csv(file.path)
                else:
                    df = pd.read_excel(file.path)
                session_store.set_dataframe(session_id, df)
                print(f"--- Loaded {file.name} into Pandas DataFrame ---")
            except Exception as e:
                print(f"Error loading tabular data: {e}")
                
        # 2. Handle Text Data (PDF, TXT, DOCX)
        elif ext in ['.pdf', '.txt', '.docx']:
            try:
                if ext == '.pdf':
                    loader = PyPDFLoader(file.path)
                elif ext == '.txt':
                    loader = TextLoader(file.path)
                elif ext == '.docx':
                    loader = UnstructuredWordDocumentLoader(file.path)
                text_docs.extend(loader.load())
                print(f"--- Extracted text from {file.name} ---")
            except Exception as e:
                print(f"Error loading text document: {e}")

    # 3. Create Ephemeral FAISS Index if we have text documents
    if text_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(text_docs)
        
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        session_store.set_vector_store(session_id, vectorstore)
        print(f"--- Created Ephemeral FAISS Index with {len(chunks)} chunks ---")