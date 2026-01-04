
# RAG Demo with agentic AI
```
mini-agentic-rag/
├── data/               # Drop your domain PDFs/text here
├── index/              # FAISS index files will be saved here
├── .env                # Azure API Keys and Endpoints
├── requirements.txt
├── main.py             # CLI Loop
├── ingestion.py        # PDF loading -> Chunking -> FAISS
├── state.py            # LangGraph state definition
└── nodes.py            # Functions for Retrieval, Generation, and Reflection
```