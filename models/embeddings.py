# Document ingestion and embedding generation for RAG Chatbot
import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2


# DocumentEmbedder for resume/job description analysis
class DocumentEmbedder:
	def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
		self.model = SentenceTransformer(model_name)

	def load_pdf(self, file_path: str) -> str:
		text = ""
		with open(file_path, 'rb') as f:
			reader = PyPDF2.PdfReader(f)
			for page in reader.pages:
				text += page.extract_text() or ""
		return text

	def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
		words = text.split()
		return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

	def embed_chunks(self, chunks: List[str]) -> np.ndarray:
		return self.model.encode(chunks)

	def process_document(self, file_path: str) -> List[dict]:
		if file_path.lower().endswith('.pdf'):
			text = self.load_pdf(file_path)
		else:
			with open(file_path, 'r', encoding='utf-8') as f:
				text = f.read()
		chunks = self.chunk_text(text)
		embeddings = self.embed_chunks(chunks)
		return [{"chunk": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]

	def extract_sections(self, text: str) -> dict:
		# Simple extraction for resume/job description sections
		sections = {}
		lines = text.split('\n')
		current = None
		for line in lines:
			l = line.strip().lower()
			if 'education' in l:
				current = 'education'
				sections[current] = []
			elif 'experience' in l or 'work history' in l:
				current = 'experience'
				sections[current] = []
			elif 'skills' in l:
				current = 'skills'
				sections[current] = []
			elif current:
				sections[current].append(line.strip())
		return {k: '\n'.join(v) for k, v in sections.items()}
