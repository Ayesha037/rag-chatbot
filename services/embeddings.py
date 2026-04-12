import os
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class EmbeddingManager:

    def __init__(self):
        self.vectorstore_path = os.getenv(
            "VECTORSTORE_PATH", "vectorstore/faiss_index"
        )
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info("(First run downloads ~90MB — please wait...)")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore: Optional[FAISS] = None
        logger.success("Embedding model loaded! (Free, local, no API key)")

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("No documents to embed!")

        logger.info(f"Embedding {len(documents)} chunks locally...")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        logger.success(f"Vector store created: {len(documents)} vectors")
        return self.vectorstore

    def save_vectorstore(self, path: str = None) -> str:
        """Save FAISS index to disk."""
        if not self.vectorstore:
            raise ValueError("No vectorstore to save!")

        save_path = path or self.vectorstore_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(save_path)
        logger.success(f"Saved to: {save_path}")
        return save_path

    def load_vectorstore(self, path: str = None) -> FAISS:
        """Load FAISS index from disk."""
        load_path = path or self.vectorstore_path

        if not Path(load_path + ".faiss").exists() and not Path(load_path).is_dir():
            raise FileNotFoundError(
                f"No vectorstore found at: {load_path}\n"
                "Please upload documents first."
            )

        self.vectorstore = FAISS.load_local(
            load_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.success("Vector store loaded!")
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if not self.vectorstore:
            self.create_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)
            logger.success(f"Added {len(documents)} chunks")

    def vectorstore_exists(self, path: str = None) -> bool:
        """Check if saved vectorstore exists on disk."""
        check_path = path or self.vectorstore_path
        return (
            Path(check_path + ".faiss").exists() or
            Path(check_path).is_dir()
        )

    def get_vectorstore(self):
        return self.vectorstore


if __name__ == "__main__":
    from services.pdf_processor import load_pdfs_from_folder
    from services.chunker import DocumentChunker

    print("\n" + "=" * 55)
    print("  PHASE 4 - EMBEDDINGS + FAISS TEST (FREE!)")
    print("=" * 55)

    print("\n[1/4] Loading PDFs...")
    docs = load_pdfs_from_folder("data")
    if not docs:
        print("No PDFs in data/ folder!")
        exit(1)

    print("[2/4] Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_multiple_documents(docs)
    langchain_docs = chunker.chunks_to_langchain_docs(chunks)
    print(f"      {len(langchain_docs)} chunks ready")

    print("[3/4] Creating embeddings (FREE local model)...")
    manager = EmbeddingManager()
    manager.create_vectorstore(langchain_docs)

    print("[4/4] Saving vector store to disk...")
    manager.save_vectorstore()

    print("\nVerifying reload works...")
    manager2 = EmbeddingManager()
    manager2.load_vectorstore()

    print("\nTesting similarity search...")
    query = "What is artificial intelligence?"
    results = manager2.vectorstore.similarity_search(query, k=2)

    print(f"\nQuery  : {query}")
    print(f"Results: {len(results)}")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] Source : {r.metadata['source']}")
        print(f"       Content: {r.page_content[:120]}")

    print("\n" + "=" * 55)
    print("  Phase 4 Complete! (Zero cost!)")
    print("=" * 55)