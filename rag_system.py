"""
BST-AMS15 Engineering Teamwork I

This module handles:
- Document loading (PDF, text, web content, video transcripts)
- Text chunking
- Embedding generation
- Vector storage and similarity search
- LLM answer generation (Ollama - FREE, LOCAL, NO API KEY)

to run: python -m streamlit run streamlit_app.py

"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass

# ==================================================
# SECTION 1: DATA CLASSES
# ==================================================

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_id: int = 0


# ==================================================
# SECTION 2: DOCUMENT LOADERS (Multiple Formats)
# ==================================================

class DocumentLoaders:
    """Load documents from various formats"""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load text from PDF file using PyMuPDF"""
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text

    @staticmethod
    def load_text(file_path: str) -> str:
        """Load plain text file"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def load_web(url: str) -> str:
        """Load clean article content from any web page"""
        import requests
        import re
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        content = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id="mw-content-text") or
            soup.find(id="content") or
            soup.find(id="main") or
            soup.body or
            soup
        )

        paragraphs = content.find_all("p")
        
        text_parts = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) < 40:
                continue
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                text_parts.append(text)

        clean_text = "\n\n".join(text_parts)

        if len(clean_text) < 200:
            text = content.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if len(line) > 40]
            clean_text = "\n".join(lines)

        if len(clean_text) < 100:
            raise RuntimeError(f"Only {len(clean_text)} chars extracted. Site may be JS-heavy.")

        print(f"  - Extracted {len(clean_text)} chars from {url}")
        return clean_text

    @staticmethod
    def load_youtube_transcript(video_url: str) -> str:
        """Load transcript from YouTube video"""
        from youtube_transcript_api import YouTubeTranscriptApi

        video_url = (video_url or "").strip()
        if not video_url:
            raise ValueError("Empty YouTube URL")

        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/shorts/" in video_url:
            video_id = video_url.split("shorts/")[1].split("?")[0]
        else:
            video_id = video_url.strip()

        try:
            api = YouTubeTranscriptApi()
            transcripts = api.list(video_id)

            transcript_obj = None
            for t in transcripts:
                lang = getattr(t, "language_code", None)
                if lang and lang.startswith("en"):
                    transcript_obj = t
                    break
            if transcript_obj is None:
                transcript_obj = next(iter(transcripts))

            items = transcript_obj.fetch()

            parts = []
            for item in items:
                parts.append(getattr(item, "text", ""))

            text = " ".join(p for p in parts if p).strip()
            if not text:
                raise RuntimeError("Transcript fetched but empty.")
            return text

        except Exception as e:
            raise RuntimeError(f"YouTube transcript failed for video_id={video_id}. Reason: {e}")


# ==================================================
# SECTION 3: TEXT CHUNKING
# ==================================================

class TextChunker:
    """Split text into overlapping chunks"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "unknown") -> List[Document]:
        """Split text into overlapping chunks"""
        chunks: List[Document] = []
        start = 0
        chunk_id = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append(Document(
                    content=chunk_content,
                    metadata={"chunk_size": len(chunk_content), "start_char": start, "end_char": end},
                    source=source,
                    chunk_id=chunk_id
                ))
                chunk_id += 1

            start = max(start + 1, end - self.chunk_overlap)

        return chunks


# ==================================================
# SECTION 4: EMBEDDINGS
# ==================================================

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers (FREE, local)"""

    def __init__(self, model_type: str = "sentence-transformers"):
        self.model_type = model_type
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_type == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown embedding model: {self.model_type}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def generate_single_embedding(self, text: str) -> List[float]:
        return self.generate_embeddings([text])[0]


# ==================================================
# SECTION 5: VECTOR DATABASE (ChromaDB)
# ==================================================

class VectorDatabase:
    """Store and search embeddings using ChromaDB"""

    def __init__(self, embedding_dim: int = 384):
        import chromadb
        from chromadb.config import Settings

        print("  - Initializing ChromaDB...")

        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

        self.documents: List[Document] = []
        self.doc_count = 0

        print("  - ChromaDB initialized!")

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        ids = [f"doc_{self.doc_count + i}" for i in range(len(documents))]
        texts = [doc.content for doc in documents]
        metadatas = [{"source": doc.source, "chunk_id": doc.chunk_id} for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        self.documents.extend(documents)
        self.doc_count += len(documents)
        print(f"  - Added {len(documents)} documents to ChromaDB")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results and results.get("documents") and results["documents"][0]:
            for doc_text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                score = 1 - distance
                doc = Document(
                    content=doc_text,
                    metadata=metadata,
                    source=metadata.get("source", "unknown"),
                    chunk_id=metadata.get("chunk_id", 0)
                )
                output.append((doc, score))

        return output

    def clear(self):
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.documents = []
        self.doc_count = 0


# ==================================================
# SECTION 6: RAG PIPELINE (Ollama - FREE, LOCAL)
# ==================================================

class RAGPipeline:
    """Complete RAG system combining all components"""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        embedding_model: str = "sentence-transformers",
        ollama_model: str = "llama3.2"
    ):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator(embedding_model)
        self.vector_db = VectorDatabase(embedding_dim=384)
        self.ollama_model = ollama_model
        self.loaders = DocumentLoaders()

    def add_document(self, source: str, source_type: str = "text"):
        """Add a document to the RAG system"""
        source_type = (source_type or "").strip().lower()

        if source_type == "pdf":
            text = self.loaders.load_pdf(source)
        elif source_type in ("text", "txt"):
            text = self.loaders.load_text(source)
        elif source_type == "web":
            text = self.loaders.load_web(source)
        elif source_type == "youtube":
            text = self.loaders.load_youtube_transcript(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        print(f"Text loaded: {len(text)} chars")

        print("Chunking text...")
        chunks = self.chunker.chunk_text(text, source)
        print(f"Created {len(chunks)} chunks from {source}")

        print("Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(chunk_texts)
        print("Embeddings generated!")

        print("Storing in vector database...")
        self.vector_db.add_documents(chunks, embeddings)
        print(f"Added {len(chunks)} chunks to vector database")

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple]:
        """Retrieve relevant chunks for a query"""
        query_embedding = self.embedder.generate_single_embedding(query)
        return self.vector_db.search(query_embedding, top_k)

    def generate_answer(self, query: str, top_k: int = 5) -> str:
        """Generate an answer using Ollama (FREE, LOCAL)"""
        import requests
        
        results = self.retrieve(query, top_k)

        if not results:
            return "I couldn't find any relevant information in the documents."

        # Build context from retrieved chunks
        context_parts = []
        for i, (doc, score) in enumerate(results):
            if score > 0.2:
                context_parts.append(f"[Source {i+1}: {doc.source}]\n{doc.content}")

        if not context_parts:
            return "I found some documents but none were relevant enough to answer your question."

        context = "\n\n---\n\n".join(context_parts)

        # Prompt template
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
- Use the context below to answer the question
- If the context contains the answer, provide a clear and comprehensive response
- If the context does NOT contain enough information, say "Based on the available documents, I don't have enough information to answer this."
- Do NOT make up information that isn't in the context
- Be concise but thorough

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # Call Ollama (LOCAL, FREE, NO API KEY)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response from Ollama")
            else:
                return f"⚠️ Ollama error: {response.text}\n\nMake sure Ollama is running: `ollama serve`"
                
        except requests.exceptions.ConnectionError:
            return f"⚠️ Cannot connect to Ollama.\n\n**To fix:**\n1. Open a new terminal\n2. Run: `ollama serve`\n3. Try again"
        except Exception as e:
            return f"⚠️ Ollama error: {e}"


# ==================================================
# SECTION 7: MAIN EXAMPLE
# ==================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RAG System - Example Usage")
    print("=" * 50)

    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        chunk_size=1200,
        chunk_overlap=150,
        embedding_model="sentence-transformers",
        ollama_model="llama3.2"
    )
    print("RAG pipeline initialized!")

    print("Loading content...")
    try:
        # PDF example
        # rag.add_document("/Users/fufu/Desktop/document.pdf", "pdf")

        # TXT example
        # rag.add_document("/Users/fufu/Desktop/adaptive_suspension.txt", "text")

        # Web example
        # rag.add_document("https://en.wikipedia.org/wiki/Louis_XIV", "web")

        # YouTube example
        rag.add_document("https://youtu.be/nC6fsNXdcMQ?si=UVvc20Z87kC9X7KM", "youtube")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Example: Query
    '''
    results = rag.retrieve("What is Python?", top_k=5)
    for doc, score in results:
        print(f"\nScore: {score:.4f}")
        print(f"Source: {doc.source} | chunk_id: {doc.chunk_id}")
        print(doc.content[:400])
    '''

    # Example: Generate answer
    '''
    answer = rag.generate_answer("What is Python?")
    print(answer)
    '''

    print("\nTo use this RAG system:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Install Ollama: brew install ollama")
    print("3. Download model: ollama pull llama3.2")
    print("4. Start Ollama: ollama serve")
    print("5. Add documents using rag.add_document()")
    print("6. Query using rag.retrieve() or rag.generate_answer()")
    print("\nOr run the web interface: streamlit run streamlit_app.py")