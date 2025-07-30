# complete_legal_rag_chatbot.py

import re
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown

class LegalRAGChatbot:
    def __init__(self):
        # Initialize embedding model
        self.embed_model_name = "all-MiniLM-L6-v2"
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model_name
        )
        # Initialize Chroma client and collection
        self.client = chromadb.Client()
        self.collection = None

        # Initialize local text generation pipeline
        from transformers import pipeline
        self.text_generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            return_full_text=False,
            pad_token_id=50256
        )
    
    def extract_legal_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract legal text chunks by detecting sections/chapters.
        Adjust regexes as per your document format.
        """
        section_title = ""
        chunks = []
        current = ""

        for line in text.splitlines():
            line = line.strip()
            # Detect section titles (simple heuristic, customize as needed)
            if re.match(r"^(CHAPTER|Section|SEC\.|ARTICLE|PART)\b", line, re.IGNORECASE):
                section_title = line
                if current:
                    chunks.append({"section": section_title, "text": current.strip()})
                    current = ""
            elif re.match(r"^\(\d+[A-Za-z]*\)", line):  # subsection/paragraph start
                if current:
                    chunks.append({"section": section_title, "text": current.strip()})
                current = line
            else:
                if current:
                    current += " " + line
                else:
                    current = line

        if current:
            chunks.append({"section": section_title, "text": current.strip()})
        return chunks

    def process_pdf(self, pdf_io) -> str:
        """
        Accepts a file-like object (Streamlit upload) for PDF processing.
        Converts PDF to text/chunks, indexes with embeddings.
        """
        try:
            md = MarkItDown()
            # Convert PDF bytes/file to markdown
            result = md.convert(pdf_io)
            text = result.text_content
            chunks = self.extract_legal_chunks(text)

            # Delete old collection (if exists)
            try:
                self.client.delete_collection(name="legal_docs")
            except Exception:
                pass

            # Create new collection with embedding function
            self.collection = self.client.create_collection(
                name="legal_docs",
                embedding_function=self.embed_fn
            )

            # Add chunks to vector DB
            for i, chk in enumerate(chunks):
                if chk["text"].strip():
                    self.collection.add(
                        documents=[chk["text"]],
                        metadatas=[{"section": chk["section"]}],
                        ids=[f"chunk_{i}"]
                    )
            return f"Successfully indexed {len(chunks)} chunks from PDF."
        except Exception as e:
            return f"PDF processing error: {e}"

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        if not self.collection:
            return "No document indexed yet. Please upload and process a PDF first."
        res = self.collection.query(query_texts=[query], n_results=n_results)
        docs = res.get("documents", [[]])[0]
        if not docs:
            return "No relevant sections found."
        return "\n\n".join(docs)

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        out = self.text_generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        return out[0]["generated_text"].strip()

    def chat(self, question: str) -> str:
        context = self.retrieve_context(question)
        answer = self.generate_response(question, context)
        return answer
