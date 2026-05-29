import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.search_agent import search_agent
from agents.summarize_agent import summarize_agent
from agents.answer_agent import answer_agent

# Shared state passed between all agents
class AgentState(TypedDict):
    query: str
    strategy: str
    facts: str
    summary: str
    answer: str
    chunks_retrieved: int   
    timings: dict
    conversation_id: str
    raw_chunks: list

# Each node wraps an agent and tracks timing
def run_search(state: AgentState) -> AgentState:
    start = time.time()
    result = search_agent(state["query"], strategy=state["strategy"])
    state["facts"] = result["facts"]
    state["chunks_retrieved"] = result["chunks_retrieved"]
    state["raw_chunks"] = result["raw_chunks"]
    state["timings"]["search"] = round(time.time() - start, 3)
    print(f"  [search] done in {state['timings']['search']}s — {state['chunks_retrieved']} chunks retrieved")
    return state

def run_summarize(state: AgentState) -> AgentState:
    start = time.time()
    result = summarize_agent(state["facts"], state["query"])
    state["summary"] = result["summary"]
    state["timings"]["summarize"] = round(time.time() - start, 3)
    print(f"  [summarize] done in {state['timings']['summarize']}s")
    return state

def run_answer(state: AgentState) -> AgentState:
    start = time.time()
    result = answer_agent(state["query"], state["summary"])
    state["answer"] = result["answer"]
    state["timings"]["answer"] = round(time.time() - start, 3)
    print(f"  [answer] done in {state['timings']['answer']}s")
    return state

# Build the graph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search", run_search)
    graph.add_node("summarize", run_summarize)
    graph.add_node("answer", run_answer)

    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "answer")
    graph.add_edge("answer", END)

    return graph.compile()

# Main entrypoint
def run_pipeline(query: str, strategy: str = "semantic", conversation_id: str = "conv_001") -> AgentState:
    graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "strategy": strategy,
        "facts": "",
        "summary": "",
        "answer": "",
        "chunks_retrieved": 0,
        "raw_chunks": [],
        "timings": {},
        "conversation_id": conversation_id
    }

    print(f"\n[supervisor] Starting pipeline | conv_id={conversation_id} | strategy={strategy}")
    result = graph.invoke(initial_state)
    total = sum(result["timings"].values())
    print(f"[supervisor] Pipeline complete | total={round(total, 3)}s\n")
    return result