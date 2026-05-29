from agents.search_agent import search_agent
from agents.summarize_agent import summarize_agent
from agents.answer_agent import answer_agent

query = "How does chunking strategy affect RAG quality?"

print("=== SEARCH AGENT ===")
search_result = search_agent(query)
print(search_result["facts"])

print("\n=== SUMMARIZE AGENT ===")
summarize_result = summarize_agent(search_result["facts"], query)
print(summarize_result["summary"])

print("\n=== ANSWER AGENT ===")
answer_result = answer_agent(query, summarize_result["summary"])
print(answer_result["answer"])