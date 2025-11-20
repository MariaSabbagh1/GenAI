import os
import re
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama  # local LLM


class RAGService:
    def __init__(self, db_path: str = "./data/chroma_db"):
        os.makedirs(db_path, exist_ok=True)

        self.db_path = db_path

        # 🔹 Local, FREE embeddings (no API key needed)
        self.embedding = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # 🔹 Local LLM via Ollama (no OpenAI quota)
        # Make sure you have ollama installed and a model pulled, e.g.:
        #   ollama pull llama3.2
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.2,
        )

    # 🔹 Helper: sanitize device_id to be valid for Chroma collection_name
    def _sanitize_device_id(self, device_id: str) -> str:
        """
        Chroma collection_name must:
        - only contain [a-zA-Z0-9._-]
        - be 3–512 chars
        - start/end with [a-zA-Z0-9]
        """

        # Replace all invalid chars with "_"
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", device_id)

        # Strip invalid start/end chars
        cleaned = cleaned.strip("._-")

        # Ensure minimum length
        if len(cleaned) < 3:
            cleaned = f"dev_{cleaned or 'manual'}"

        # Max 512 chars
        return cleaned[:512]

    def _get_vectorstore(self, device_id: str) -> Chroma:
        """
        Create or connect to a Chroma collection for this device_id.
        We sanitize the ID to be safe for Chroma.
        """
        safe_device_id = self._sanitize_device_id(device_id)

        return Chroma(
            collection_name=safe_device_id,
            embedding_function=self.embedding,
            persist_directory=self.db_path,
        )

    def index_manual(self, device_id: str, file_bytes: bytes, filename: str) -> int:
        """
        Index a PDF manual for a given device_id.
        Returns number of chunks stored.
        """
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load PDF as documents
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()  # each doc has page_content + metadata (incl. page)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Prepare vectorstore and add documents
        vectorstore = self._get_vectorstore(device_id)
        vectorstore.add_documents(chunks)
        vectorstore.persist()

        # Remove temp file
        os.remove(tmp_path)

        return len(chunks)

    def ask_question(self, device_id: str, question: str) -> Tuple[str, List[dict]]:
        """
        Answer a question based on the indexed manual for the given device_id.
        Returns (answer, sources).
        """
        vectorstore = self._get_vectorstore(device_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # New API: retriever.invoke instead of get_relevant_documents
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return (
                "I couldn't find any relevant information in the manual for this question.",
                [],
            )

        # Build context
        context_parts = []
        sources = []
        for doc in relevant_docs:
            text = doc.page_content
            page = doc.metadata.get("page", None)
            context_parts.append(text)
            sources.append(
                {
                    "page": int(page) + 1 if isinstance(page, int) else None,
                    "snippet": text[:300] + ("..." if len(text) > 300 else ""),
                }
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are an assistant helping a user with an appliance manual.\n"
            "You must ONLY use the information from the manual context below.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"MANUAL CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Provide a clear, step-by-step answer. Mention any warnings if they appear."
        )

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return answer, sources

