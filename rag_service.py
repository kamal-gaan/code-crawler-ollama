# rag_service.py

import os
import pathspec
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_unstructured import UnstructuredLoader

# At the top of rag_service.py
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import settings from our config file
import config


class RAGService:
    def __init__(self, collection_name="default"):
        """
        Initializes the RAG service for a specific collection.
        A collection corresponds to an indexed codebase.
        """
        self.collection_name = collection_name
        self.persist_path = os.path.join(config.PERSIST_DIRECTORY, self.collection_name)

        self.llm = OllamaLLM(
            model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0
        )
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL
        )
        self.vectorstore = self._load_vectorstore()
        self.rag_chain = self._create_rag_chain()

    def _load_vectorstore(self):
        """Loads a persistent Chroma vector store if it exists, otherwise returns None."""
        if os.path.exists(self.persist_path):
            print(f"Loading existing vector store from: {self.persist_path}")
            return Chroma(
                persist_directory=self.persist_path, embedding_function=self.embeddings
            )
        print("No existing vector store found.")
        return None

    def _format_docs(self, docs):
        """Helper function to format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_rag_chain(self):
        """Creates the LangChain RAG chain if the vector store is available."""
        if not self.vectorstore:
            return None

        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.RETRIEVER_K}
        )

        template = """
        You are an expert programmer and assistant. Answer the user's question based ONLY on the following context of code snippets.
        If you don't know the answer from the context provided, just say that you don't know. Do not make up an answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(template)

        return (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # In rag_service.py, inside the RAGService class

    def index_codebase(self, code_path: str):
        # ... (The file finding logic is unchanged)
        gitignore_path = os.path.join(code_path, ".gitignore")
        files_to_load = []
        if os.path.exists(gitignore_path):
            print("Found .gitignore, applying ignore rules...")
            with open(gitignore_path, "r") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
            for root, dirs, files in os.walk(code_path, topdown=True):
                if ".git" in dirs:
                    dirs.remove(".git")
                dirs[:] = [
                    d
                    for d in dirs
                    if not spec.match_file(
                        os.path.relpath(os.path.join(root, d), code_path)
                    )
                ]
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, code_path)
                    if not spec.match_file(relative_path):
                        files_to_load.append(full_path)
        else:
            print("No .gitignore found. Loading all files based on glob pattern.")
            fallback_loader = DirectoryLoader(
                code_path, glob=config.TARGET_FILE_GLOB, recursive=True
            )
            files_to_load = [doc.metadata["source"] for doc in fallback_loader.load()]

        print(f"Found {len(files_to_load)} files to load after filtering.")
        if not files_to_load:
            raise ValueError("No files found to index after applying filters.")

        print("Loading documents from the filtered file list...")
        docs = []
        for file_path in files_to_load:
            try:
                loader = UnstructuredLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Could not load file {file_path}. Error: {e}")

        if not docs:
            raise ValueError("Document loading resulted in zero documents.")

        # ... (The summarization logic is unchanged)
        summaries = []
        if config.SUMMARIZE_DOCUMENTS:
            print(
                f"Summarization is ENABLED. Generating summaries for {len(docs)} documents... (This will be slow)"
            )
            summary_template = "..."  # Your summary template
            summary_prompt = ChatPromptTemplate.from_template(summary_template)
            summarizer_chain = summary_prompt | self.llm | StrOutputParser()
            for doc in docs:
                summary = summarizer_chain.invoke({"file_content": doc.page_content})
                summary_doc = Document(
                    page_content=summary,
                    metadata={"source": doc.metadata["source"], "type": "summary"},
                )
                summaries.append(summary_doc)
        else:
            print("Summarization is DISABLED for faster indexing.")

        for doc in docs:
            doc.metadata["type"] = "code"

        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)

        print("Creating and persisting vector store...")
        os.makedirs(self.persist_path, exist_ok=True)

        docs_to_index = splits + summaries

        # --- THIS IS THE FIX ---
        # ChromaDB requires simple metadata. This utility removes complex values like lists.
        print("Filtering complex metadata from documents...")
        filtered_docs = filter_complex_metadata(docs_to_index)
        # --- END OF FIX ---

        self.vectorstore = Chroma.from_documents(
            documents=filtered_docs,  # <-- Use the new filtered list here
            embedding=self.embeddings,
            persist_directory=self.persist_path,
        )
        self.rag_chain = self._create_rag_chain()
        print("Indexing complete!")
        return len(docs)

    def ask_question(self, question: str):
        """
        Asks a question to the RAG chain.
        """
        if not self.rag_chain:
            raise ValueError(
                "The RAG chain is not initialized. Please index a codebase first."
            )

        print(f"Invoking chain with question: '{question}'")
        return self.rag_chain.invoke(question)
