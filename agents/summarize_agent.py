from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_agent(facts: str, query: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a summarization agent. Condense facts into a clear concise summary. Remove redundancy. Keep only what matters."},
            {"role": "user", "content": f"Original Query: {query}\n\nFacts to summarize:\n{facts}\n\nConcise summary:"}
        ]
    )

    return {
        "agent": "summarize",
        "summary": response.choices[0].message.content
    }