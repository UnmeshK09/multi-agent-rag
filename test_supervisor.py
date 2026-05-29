from agents.supervisor import run_pipeline

result = run_pipeline(
    query="What is RAG and why does chunking strategy matter?",
    strategy="semantic",
    conversation_id="conv_001"
)

print("=" * 50)
print("FINAL ANSWER")
print("=" * 50)
print(result["answer"])
print("\nTIMINGS:", result["timings"])
print("CONV ID:", result["conversation_id"])