import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from loguru import logger

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

from services.retriever import RetrieverService

load_dotenv()

SYSTEM_PROMPT = """You are an expert document analyst and helpful assistant.
Your job is to provide detailed, accurate answers based on the provided context.

RULES:
1. Answer ONLY from the provided context
2. Give DETAILED and COMPLETE answers — do not be too brief
3. If asked for a summary, provide a proper 3-5 sentence paragraph summary
4. If asked to explain something, explain it clearly and fully
5. Always structure your answer well with proper sentences
6. If the answer is truly not in the context, say exactly:
   "I could not find information about this in the provided documents."
7. Never make up information outside the context

ANSWER STYLE:
- Be comprehensive but concise
- Use bullet points for lists
- Use paragraphs for explanations
- Always sound professional and helpful
- Give complete answers, not one-liners
"""


class RAGPipeline:
    """
    The complete RAG pipeline.
    Combines retrieval + prompt engineering + LLM generation.

    Usage:
        pipeline = RAGPipeline(vectorstore)
        result = pipeline.query("What is AI?")
        print(result["answer"])
        print(result["citations"])
    """

    def __init__(
        self,
        vectorstore: FAISS,
        top_k: int = None,
        model_name: str = None,
    ):
        self.retriever = RetrieverService(
            vectorstore=vectorstore,
            top_k=top_k or int(os.getenv("TOP_K_RESULTS", "5"))
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key.startswith("gsk_your"):
            raise ValueError(
                "GROQ_API_KEY not set!\n"
                "Get a free key at: https://console.groq.com"
            )

        self.model_name = model_name or os.getenv(
            "LLM_MODEL", "llama-3.1-8b-instant"
        )

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=2048,
        )

        self.chat_history: List[Dict] = []

        logger.info(
            f"RAGPipeline ready: model={self.model_name}, "
            f"top_k={self.retriever.top_k}"
        )

    def query(
        self,
        question: str,
        use_history: bool = True
    ) -> Dict:
        """
        Main query method — the full RAG pipeline.

        Steps:
        1. Retrieve relevant chunks from FAISS
        2. Format context with citations
        3. Build detailed prompt
        4. Call Groq LLM
        5. Return answer + citations
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty!")

        logger.info(f"Processing query: '{question[:60]}'")

        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:
            return {
                "answer": (
                    "I could not find any relevant information "
                    "in the uploaded documents. Please make sure "
                    "you have uploaded a PDF and try again."
                ),
                "context": "",
                "citations": [],
                "question": question,
                "model": self.model_name
            }

        context = self.retriever.format_context(retrieved_docs)
        citations = self.retriever.get_source_citations(retrieved_docs)

        prompt = self._build_prompt(
            question=question,
            context=context,
            use_history=use_history
        )

        logger.info(f"Calling {self.model_name} via Groq...")
        answer = self._call_llm(prompt)

        self.chat_history.append({
            "role": "user",
            "content": question
        })
        self.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        logger.success(f"Answer generated: {len(answer)} chars")

        return {
            "answer": answer,
            "context": context,
            "citations": citations,
            "question": question,
            "model": self.model_name
        }

    def _build_prompt(
        self,
        question: str,
        context: str,
        use_history: bool = True
    ) -> str:
        """
        Build a detailed, well-structured prompt for the LLM.

        Structure:
        - Context from documents
        - Recent conversation history
        - User question
        - Clear instructions for detailed answers
        """

        history_text = ""
        if use_history and self.chat_history:
            recent = self.chat_history[-6:]
            lines = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content'][:300]}")
            history_text = (
                "\n\nPREVIOUS CONVERSATION (for context):\n" +
                "\n".join(lines)
            )

        prompt = f"""You have been provided with the following context
extracted from the uploaded document(s). Use this context to answer
the user's question thoroughly and accurately.

════════════════════════════════════════
DOCUMENT CONTEXT:
════════════════════════════════════════
{context}
════════════════════════════════════════
{history_text}

USER QUESTION: {question}

════════════════════════════════════════
INSTRUCTIONS FOR YOUR ANSWER:
════════════════════════════════════════
- Read the context carefully above
- Provide a DETAILED and COMPLETE answer
- If asked for a summary: write 3-5 full sentences
- If asked to explain: give a thorough explanation
- If asked for key points: use bullet points
- Base your answer ENTIRELY on the context provided
- If the information is not in the context, say so clearly
- Do NOT give one-line answers — be comprehensive
- Mention the source document when relevant

YOUR DETAILED ANSWER:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the Groq LLM with system + user messages."""
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")

    def get_history(self) -> List[Dict]:
        """Return full chat history."""
        return self.chat_history

    def get_history_as_string(self) -> str:
        """Return chat history as formatted string."""
        if not self.chat_history:
            return "No conversation history yet."
        lines = []
        for msg in self.chat_history:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)


def create_rag_pipeline(vectorstore: FAISS) -> RAGPipeline:
    """Create a RAGPipeline from an existing vectorstore."""
    return RAGPipeline(vectorstore=vectorstore)


if __name__ == "__main__":
    from services.embeddings import EmbeddingManager

    print("\n" + "=" * 55)
    print("  PHASE 6 - RAG PIPELINE TEST")
    print("=" * 55)

    print("\n[1/3] Loading vector store...")
    try:
        manager = EmbeddingManager()
        manager.load_vectorstore()
        print("      Vector store loaded!")
    except FileNotFoundError:
        print("      Run Phase 4 first!")
        exit(1)

    print("[2/3] Creating RAG pipeline...")
    try:
        pipeline = RAGPipeline(vectorstore=manager.get_vectorstore())
        print("      Pipeline ready!")
    except ValueError as e:
        print(f"      Error: {e}")
        exit(1)

    print("[3/3] Testing RAG pipeline...\n")

    test_questions = [
        "Give me a detailed summary of this document",
        "What are the main topics covered?",
        "What is the capital of France?"
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        print("-" * 55)
        result = pipeline.query(question)
        print(f"A: {result['answer']}")
        print(f"Citations: {result['citations']}")

    print("\n" + "=" * 55)
    print("  Phase 6 Complete!")
    print("=" * 55)