import chromadb
from chromadb.utils import embedding_functions

def get_retriever(strategy: str = "semantic"):
    client = chromadb.PersistentClient(path=f"data/chroma_{strategy}")
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_collection(
        name=f"knowledge_base_{strategy}",
        embedding_function=ef
    )
    
    return collection

def retrieve(query: str, strategy: str = "semantic", n_results: int = 3) -> list[dict]:
    collection = get_retriever(strategy)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i]["source"],
            "strategy": strategy
        })
    
    return chunks