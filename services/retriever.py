import os
from typing import List, Dict, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from loguru import logger


class RetrieverService:
    """
    Handles similarity search against the FAISS vector store.

    This is the core of RAG — finding the right context
    to answer a user's question accurately.

    Key concept: We don't search by keywords.
    We search by MEANING using vector similarity.
    
    Example:
        Query: "How does ML work?"
        Finds: chunk about "machine learning algorithms"
        Even though "ML" != "machine learning" — same meaning!
    """

    def __init__(self, vectorstore: FAISS, top_k: int = None):
        """
        Initialize with a loaded FAISS vectorstore.

        Args:
            vectorstore: Loaded FAISS index from EmbeddingManager
            top_k: Number of chunks to retrieve per query
                   Default from .env or 5
        """
        if vectorstore is None:
            raise ValueError(
                "Vectorstore is None! "
                "Please load or create a vectorstore first."
            )

        self.vectorstore = vectorstore
        self.top_k = top_k or int(os.getenv("TOP_K_RESULTS", "5"))

        logger.info(
            f"RetrieverService ready: top_k={self.top_k}"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.

        This is the simplest retrieval method —
        pure vector similarity search.

        Args:
            query: User's question as plain text

        Returns:
            List of LangChain Document objects,
            sorted by relevance (most relevant first)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty!")

        logger.info(f"Retrieving chunks for: '{query[:50]}...'")

        results = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k
        )

        logger.info(f"Retrieved {len(results)} chunks")
        return results

    def retrieve_with_scores(
        self, query: str
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve chunks WITH their similarity scores.

        Scores are cosine similarity (0 to 1):
        - 1.0 = perfect match
        - 0.8+ = very relevant
        - 0.5  = somewhat relevant
        - 0.0  = completely different

        Useful for debugging and filtering low-quality results.

        Returns:
            List of (Document, score) tuples
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty!")

        logger.info(
            f"Retrieving with scores for: '{query[:50]}'"
        )

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.top_k
        )

        for doc, score in results:
            logger.debug(
                f"Score: {score:.4f} | "
                f"Source: {doc.metadata.get('source', 'unknown')} | "
                f"Preview: {doc.page_content[:50]}..."
            )

        return results

    def retrieve_with_filter(
        self,
        query: str,
        source_filter: str = None,
        min_score: float = 0.0
    ) -> List[Document]:
        """
        Retrieve chunks with optional filtering.

        Args:
            query: User question
            source_filter: Only return chunks from this PDF file
            min_score: Minimum similarity score (0.0 to 1.0)
                      Use 0.3+ to filter out irrelevant results

        Returns:
            Filtered list of relevant Documents
        """
        results_with_scores = self.retrieve_with_scores(query)

        filtered = []
        for doc, score in results_with_scores:
            if score < min_score:
                logger.debug(
                    f"Filtered out low-score chunk: {score:.4f}"
                )
                continue
            if source_filter:
                doc_source = doc.metadata.get("source", "")
                if source_filter.lower() not in doc_source.lower():
                    continue

            filtered.append(doc)

        logger.info(
            f"After filtering: {len(filtered)}/{self.top_k} chunks"
        )
        return filtered

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.

        This formatted string is what we pass to the LLM
        in Phase 6. The LLM uses this as its reference
        to answer the user's question.

        Format:
            [Source: file.pdf | Page: 1]
            chunk text here...

            [Source: file.pdf | Page: 2]
            more chunk text...

        Returns:
            Single formatted string with all context
        """
        if not documents:
            return "No relevant context found."

        context_parts = []

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_num", "?")
            chunk_id = doc.metadata.get("chunk_id", i)

            chunk_text = (
                f"[Source: {source} | Page: {page} | "
                f"Chunk: {chunk_id}]\n"
                f"{doc.page_content}"
            )
            context_parts.append(chunk_text)

        context = "\n\n---\n\n".join(context_parts)
        return context

    def get_source_citations(
        self, documents: List[Document]
    ) -> List[Dict]:
        """
        Extract source citations from retrieved documents.

        Used in the frontend to show users WHERE
        the answer came from — builds trust and
        lets users verify the information.

        Returns:
            List of citation dicts with source info
        """
        citations = []
        seen = set()  

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_num", 1)
            citation_key = f"{source}_{page}"

            if citation_key not in seen:
                citations.append({
                    "source": source,
                    "page": page,
                    "preview": doc.page_content[:100] + "..."
                })
                seen.add(citation_key)

        return citations

if __name__ == "__main__":
    from services.embeddings import EmbeddingManager

    print("\n" + "=" * 55)
    print("  PHASE 5 - RETRIEVAL SYSTEM TEST")
    print("=" * 55)

    print("\n[1/3] Loading vector store...")
    try:
        manager = EmbeddingManager()
        manager.load_vectorstore()
        print("      Vector store loaded!")
    except FileNotFoundError:
        print("      No vectorstore found!")
        print("      Run Phase 4 first: python -m services.embeddings")
        exit(1)

    print("[2/3] Creating retriever...")
    retriever = RetrieverService(
        vectorstore=manager.get_vectorstore(),
        top_k=3
    )

    print("[3/3] Testing retrieval with different queries...")

    test_queries = [
        "What is artificial intelligence?",
        "Tell me about LangChain",
        "How does FAISS work?",
        "What is this document about?"
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: {query}")
        print("-" * 55)

        results = retriever.retrieve(query)

        if not results:
            print("No results found!")
            continue

        for i, doc in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"  Source  : {doc.metadata.get('source')}")
            print(f"  Page    : {doc.metadata.get('page_num')}")
            print(f"  Content : {doc.page_content[:120]}")

        print(f"\n  Formatted Context:")
        print(f"  {'-'*40}")
        context = retriever.format_context(results)
        print(context[:300])

        citations = retriever.get_source_citations(results)
        print(f"\n  Citations: {citations}")

    print("\n" + "=" * 55)
    print("  Phase 5 Complete!")
    print("=" * 55)