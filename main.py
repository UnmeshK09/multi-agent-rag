import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from agents.supervisor import run_pipeline
from evals.pipeline import compare_strategies

console = Console()

def run_query(query: str, strategy: str):
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}\n[bold cyan]Strategy:[/bold cyan] {strategy}", title="Multi-Agent RAG"))

    result = run_pipeline(query=query, strategy=strategy, conversation_id="main_001")

    console.print(Panel(result["answer"], title="[bold green]Final Answer[/bold green]"))

    table = Table(title="Pipeline Timings")
    table.add_column("Agent", style="cyan")
    table.add_column("Time (s)", style="magenta")
    for agent, t in result["timings"].items():
        table.add_row(agent, str(t))
    console.print(table)

    console.print(f"[dim]Conversation ID: {result['conversation_id']} | Chunks retrieved: {result['chunks_retrieved']}[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG System")
    parser.add_argument("--mode", choices=["query", "eval"], default="query")
    parser.add_argument("--query", type=str, default="What is retrieval augmented generation?")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="semantic")
    args = parser.parse_args()

    if args.mode == "eval":
        compare_strategies()
    else:
        run_query(args.query, args.strategy)

if __name__ == "__main__":
    main()