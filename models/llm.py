import numpy as np
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Semantic search and LLM answer generation for RAG Chatbot
def semantic_search(query: str, embedded_docs: List[Dict], embedder) -> List[Dict]:
    query_emb = embedder.model.encode([query])[0]
    scores = [np.dot(doc['embedding'], query_emb) / (np.linalg.norm(doc['embedding']) * np.linalg.norm(query_emb)) for doc in embedded_docs]
    top_indices = np.argsort(scores)[::-1][:3]
    return [embedded_docs[i] for i in top_indices]

class LLMAnswerer:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        self.model = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant"
        )

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        context = "\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()