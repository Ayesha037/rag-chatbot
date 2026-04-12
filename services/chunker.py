import os
from typing import List, Dict
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger

from services.pdf_processor import ProcessedDocument


@dataclass
class TextChunk:
    text: str
    chunk_id: int
    source: str
    page_num: int
    total_chunks: int


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk_document(self, document: ProcessedDocument) -> List[TextChunk]:
        if not document.full_text.strip():
            logger.warning(f"Empty document: {document.filename}")
            return []

        logger.info(
            f"Chunking '{document.filename}' "
            f"({len(document.full_text):,} chars)"
        )

        raw_chunks = self.splitter.split_text(document.full_text)
        chunks = []

        for i, chunk_text in enumerate(raw_chunks):
            page_num = self._estimate_page(chunk_text, document)
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=i,
                source=document.filename,
                page_num=page_num,
                total_chunks=len(raw_chunks)
            )
            chunks.append(chunk)

        logger.success(
            f"Chunked '{document.filename}': "
            f"{len(chunks)} chunks from "
            f"{document.total_pages} page(s)"
        )
        return chunks

    def chunk_multiple_documents(
        self, documents: List[ProcessedDocument]
    ) -> List[TextChunk]:
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk {doc.filename}: {e}")
                continue

        logger.info(
            f"Total chunks from {len(documents)} document(s): "
            f"{len(all_chunks)}"
        )
        return all_chunks

    def chunks_to_langchain_docs(
        self, chunks: List[TextChunk]
    ) -> List[Document]:
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "page_num": chunk.page_num,
                    "total_chunks": chunk.total_chunks,
                }
            )
            docs.append(doc)
        return docs

    def _estimate_page(
        self, chunk_text: str, document: ProcessedDocument
    ) -> int:
        search_key = chunk_text[:100].strip()
        for page in document.pages:
            if search_key in page["text"]:
                return page["page_num"]
        return 1


if __name__ == "__main__":
    from services.pdf_processor import load_pdfs_from_folder

    print("\n" + "=" * 55)
    print("  PHASE 3 - TEXT CHUNKING TEST")
    print("=" * 55)

    docs = load_pdfs_from_folder("data")
    if not docs:
        print("No PDFs found in data/ folder!")
        exit(1)

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    all_chunks = chunker.chunk_multiple_documents(docs)
    langchain_docs = chunker.chunks_to_langchain_docs(all_chunks)

    print(f"\nChunking Results:")
    print(f"  Documents processed : {len(docs)}")
    print(f"  Total chunks        : {len(all_chunks)}")
    print(f"  LangChain docs      : {len(langchain_docs)}")

    print(f"\nFirst 3 chunks preview:")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"  Source  : {chunk.source}")
        print(f"  Page    : {chunk.page_num}")
        print(f"  Length  : {len(chunk.text)} chars")
        print(f"  Preview : {chunk.text[:150]}...")

    print(f"\nChunk size stats:")
    sizes = [len(c.text) for c in all_chunks]
    print(f"  Min : {min(sizes)} chars")
    print(f"  Max : {max(sizes)} chars")
    print(f"  Avg : {sum(sizes) // len(sizes)} chars")
    print("\n" + "=" * 55)