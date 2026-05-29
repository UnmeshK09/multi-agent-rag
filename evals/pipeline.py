import time
from agents.supervisor import run_pipeline

# Ground truth QA pairs — what a correct answer must contain
EVAL_QUESTIONS = [
    {
        "id": "eval_001",
        "query": "What is retrieval augmented generation?",
        "expected_keywords": ["retrieval", "documents", "knowledge base", "hallucination", "vector"],
        "expected_sources": ["rag_overview.txt"]
    },
    {
        "id": "eval_002",
        "query": "How does chunking strategy affect RAG quality?",
        "expected_keywords": ["chunking", "fixed", "semantic", "context", "hallucination"],
        "expected_sources": ["rag_overview.txt"]
    },
    {
        "id": "eval_003",
        "query": "What is a multi agent system?",
        "expected_keywords": ["supervisor", "agent", "tools", "ReAct", "routing"],
        "expected_sources": ["multi_agent_systems.txt"]
    },
    {
        "id": "eval_004",
        "query": "What is fine tuning an LLM?",
        "expected_keywords": ["fine-tuning", "pre-trained", "domain", "training"],
        "expected_sources": ["llm_basics.txt"]
    }
]

def score_answer(answer: str, expected_keywords: list[str]) -> float:
    """Keyword coverage score: how many expected terms appear in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 2)

def score_sources(raw_chunks: list[dict], expected_sources: list[str]) -> float:
    """Source accuracy: did retrieval pull from the right documents?"""
    retrieved_sources = set(c["source"] for c in raw_chunks)
    hits = sum(1 for s in expected_sources if s in retrieved_sources)
    return round(hits / len(expected_sources), 2)

def run_eval(strategy: str) -> list[dict]:
    print(f"\n{'='*50}")
    print(f"RUNNING EVALS — strategy: {strategy.upper()}")
    print(f"{'='*50}")

    results = []

    for q in EVAL_QUESTIONS:
        start = time.time()
        pipeline_result = run_pipeline(
            query=q["query"],
            strategy=strategy,
            conversation_id=q["id"]
        )
        latency = round(time.time() - start, 3)

        keyword_score = score_answer(pipeline_result["answer"], q["expected_keywords"])
        source_score = score_sources(pipeline_result["raw_chunks"] if "raw_chunks" in pipeline_result else [], q["expected_sources"])

        result = {
            "id": q["id"],
            "query": q["query"],
            "strategy": strategy,
            "keyword_score": keyword_score,
            "source_score": source_score,
            "latency_s": latency,
            "chunks_retrieved": pipeline_result["chunks_retrieved"]
        }

        print(f"\n  Query: {q['query']}")
        print(f"  Keyword coverage : {keyword_score:.0%}")
        print(f"  Source accuracy  : {source_score:.0%}")
        print(f"  Latency          : {latency}s")

        results.append(result)

    return results

def compare_strategies():
    fixed_results = run_eval("fixed")
    semantic_results = run_eval("semantic")

    print(f"\n{'='*50}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*50}")

    def avg(results, key):
        return round(sum(r[key] for r in results) / len(results), 2)

    print(f"\n{'Metric':<25} {'Fixed':>10} {'Semantic':>10}")
    print("-" * 45)
    print(f"{'Avg keyword coverage':<25} {avg(fixed_results, 'keyword_score'):>10.0%} {avg(semantic_results, 'keyword_score'):>10.0%}")
    print(f"{'Avg source accuracy':<25} {avg(fixed_results, 'source_score'):>10.0%} {avg(semantic_results, 'source_score'):>10.0%}")
    print(f"{'Avg latency (s)':<25} {avg(fixed_results, 'latency_s'):>10} {avg(semantic_results, 'latency_s'):>10}")
    print(f"{'Avg chunks retrieved':<25} {avg(fixed_results, 'chunks_retrieved'):>10} {avg(semantic_results, 'chunks_retrieved'):>10}")
    print()

if __name__ == "__main__":
    compare_strategies()