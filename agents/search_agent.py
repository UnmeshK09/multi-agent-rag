from groq import Groq
import os
from dotenv import load_dotenv
from rag.retriever import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_agent(query: str, strategy: str = "semantic") -> dict:
    chunks = retrieve(query, strategy=strategy)
    context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in chunks])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a search agent. Identify the most relevant facts from the context. Be concise and factual. Do not add information not present in the context."},
            {"role": "user", "content": f"Query: {query}\n\nRetrieved Context:\n{context}\n\nList the key relevant facts:"}
        ]
    )

    return {
        "agent": "search",
        "query": query,
        "chunks_retrieved": len(chunks),
        "facts": response.choices[0].message.content,
        "raw_chunks": chunks
    }