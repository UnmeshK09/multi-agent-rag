import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "data/docs"

# Two chunking strategies we'll compare in evals
CHUNKING_STRATEGIES = {
    "fixed": RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", " "]
    ),
    "semantic": RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=[". ", "! ", "? ", "\n\n", "\n"]  # splits at sentence boundaries
    )
}

def load_docs():
    docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, filename), "r") as f:
                docs.append({"text": f.read(), "source": filename})
    return docs

def build_collection(strategy_name: str):
    client = chromadb.PersistentClient(path=f"data/chroma_{strategy_name}")
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Delete if exists so we can re-run cleanly
    try:
        client.delete_collection(f"knowledge_base_{strategy_name}")
    except:
        pass
    
    collection = client.create_collection(
        name=f"knowledge_base_{strategy_name}",
        embedding_function=ef
    )
    
    splitter = CHUNKING_STRATEGIES[strategy_name]
    docs = load_docs()
    
    all_chunks, all_ids, all_metadata = [], [], []
    
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['source']}_{strategy_name}_{i}")
            all_metadata.append({"source": doc["source"], "strategy": strategy_name})
    
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
    print(f"[{strategy_name}] Indexed {len(all_chunks)} chunks from {len(docs)} docs")
    return collection

def build_all():
    for strategy in CHUNKING_STRATEGIES:
        build_collection(strategy)

if __name__ == "__main__":
    build_all()