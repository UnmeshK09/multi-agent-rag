from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_agent(query: str, summary: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a final answer agent. Give a helpful well-structured answer using only the provided summary. If the summary lacks enough information, say so clearly rather than guessing."},
            {"role": "user", "content": f"User Question: {query}\n\nSummary of retrieved knowledge:\n{summary}\n\nFinal Answer:"}
        ]
    )

    return {
        "agent": "answer",
        "answer": response.choices[0].message.content
    }