# Multi-Agent RAG System

A production-pattern retrieval-augmented generation (RAG) pipeline using LangGraph, ChromaDB, and Groq (Llama 3). A supervisor agent orchestrates three specialized sub-agents across a typed state graph, with a structured eval pipeline that benchmarks chunking strategies against real metrics.

## Architecture

User Query
↓
Supervisor Agent (LangGraph state graph)
├── Search Agent     — retrieves chunks from ChromaDB via cosine similarity
├── Summarize Agent  — condenses retrieved facts
└── Answer Agent     — synthesizes final structured response
↓
Eval Pipeline — scores keyword coverage, source accuracy, and latency

## Key Features

- **Multi-agent orchestration** via LangGraph with typed shared state and conversation ID tracing across all agent hops
- **Two RAG chunking strategies** compared head-to-head: fixed-size (200 token, 20 overlap) vs semantic (sentence-boundary-aware, 400 token, 80 overlap)
- **Structured eval pipeline** — 4 questions × 2 strategies = 8 scored runs across keyword coverage, source accuracy, and latency
- **Vector retrieval** using ChromaDB with sentence-transformers (MiniLM-L6-v2) embeddings

## Eval Results

| Metric                | Fixed  | Semantic |
|-----------------------|--------|----------|
| Avg keyword coverage  | 64%    | 79%      |
| Avg source accuracy   | 100%   | 100%     |
| Avg latency (s)       | 5.14   | 2.51     |
| Avg chunks retrieved  | 3.0    | 3.0      |

**Finding:** Semantic chunking outperformed fixed chunking by 15 percentage points on keyword coverage at half the average latency. Fixed chunking splits at token boundaries, breaking sentence context and losing keywords mid-chunk. Semantic chunking preserves natural language boundaries, resulting in higher-quality retrieval.

## Stack

- **Orchestration:** LangGraph
- **LLM:** Groq API (Llama 3 8B)
- **Vector store:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Text splitting:** LangChain text splitters

## Setup

```bash
git clone https://github.com/UnmeshK09/multi-agent-rag
cd multi-agent-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Add a `.env` file: 
GROQ_API_KEY=your_key_here

Build the knowledge base:
```bash
python rag/ingest.py
```

## Usage

```bash
# Ask a question
python main.py --mode query --query "How do multi agent systems work?" --strategy semantic

# Run full eval comparison
python main.py --mode eval
```

## Known Optimization

The sentence-transformer model currently initializes per query, causing a ~14s cold start on the first request. In production this would be initialized once at startup and held in memory, reducing search latency to ~0.4s as seen on subsequent calls.