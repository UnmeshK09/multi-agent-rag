from rag.retriever import retrieve

results = retrieve("What is retrieval augmented generation?", strategy="semantic")
for r in results:
    print(f"SOURCE: {r['source']}")
    print(f"CHUNK: {r['text'][:120]}...")
    print("---")