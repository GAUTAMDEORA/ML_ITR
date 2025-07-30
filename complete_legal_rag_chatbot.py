# complete_legal_rag_chatbot.py

import os
import re
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown

class LegalRAGChatbot:
    def __init__(self, local: bool = False):
        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # ChromaDB setup
        self.client = chromadb.Client()
        self.collection = None
        # Text-generation pipeline (local only)
        from transformers import pipeline
        self.text_generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            return_full_text=False,
            pad_token_id=50256
        )

    def extract_legal_chunks(self, text: str) -> List[Dict[str, str]]:
        section_title = ""
        chunks = []
        current = ""
        for line in text.splitlines():
            if re.match(r"^\s*CHAPTER", line) or line.strip().endswith("."):
                section_title = line.strip()
            elif re.match(r"^\(\d+[A-Z]*\)", line.strip()):
                if current:
                    chunks.append({"section": section_title, "text": current.strip()})
                current = line.strip()
            elif current:
                current += " " + line.strip()
        if current:
            chunks.append({"section": section_title, "text": current.strip()})
        return chunks

    def process_pdf(self, pdf_path: str) -> str:
        try:
            md = MarkItDown()
            result = md.convert(pdf_path)
            text = result.text_content
            chunks = self.extract_legal_chunks(text)
            try:
                self.client.delete_collection(name="legal_docs")
            except:
                pass
            self.collection = self.client.create_collection(
                name="legal_docs", embedding_function=self.embed_fn
            )
            for i, chk in enumerate(chunks):
                if chk["text"].strip():
                    self.collection.add(
                        documents=[chk["text"]],
                        metadatas=[{"section": chk["section"]}],
                        ids=[f"chk_{i}"]
                    )
            return f"Indexed {len(chunks)} chunks."
        except Exception as e:
            return f"Error: {e}"

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        if not self.collection:
            return "No document indexed."
        res = self.collection.query(query_texts=[query], n_results=n_results)
        docs = res["documents"][0]
        return "\n\n".join(docs) if docs else "No relevant sections found."

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        out = self.text_generator(
            prompt, max_length=200, num_return_sequences=1, temperature=0.7
        )
        return out[0]["generated_text"].strip()

    def chat(self, question: str) -> str:
        ctx = self.retrieve_context(question)
        return self.generate_response(question, ctx)
