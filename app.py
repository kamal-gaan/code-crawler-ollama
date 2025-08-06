import os
from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Basic Flask App Setup ---
app = Flask(__name__)

# --- Global Variables ---
# Using a simple in-memory store for the vector DB. For production, you'd persist this.
vectorstore = None
# Define the model to use. Make sure you have pulled this model with Ollama.
MODEL_NAME = "codellama"


# --- Helper Functions ---
def format_docs(docs):
    """Helper function to format documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- Flask Routes ---


@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/index_code", methods=["POST"])
def index_code():
    """
    API endpoint to index a directory of code.
    Expects a JSON payload with a 'path' key.
    """
    global vectorstore

    data = request.get_json()
    if not data or "path" not in data:
        return jsonify({"error": "Path is required"}), 400

    code_path = data["path"]
    if not os.path.isdir(code_path):
        return jsonify({"error": "Invalid path specified"}), 400

    try:
        # 1. Load Documents
        # Using a generic loader for python files. You can add more glob patterns.
        print(f"Loading documents from {code_path}...")
        loader = DirectoryLoader(code_path, glob="**/*.py", recursive=True)
        docs = loader.load()

        if not docs:
            return jsonify({"error": "No Python files found to index."}), 400

        # 2. Split Documents into Chunks
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store
        print("Creating embeddings and vector store...")
        # Note: The first time you run this, it will download the embedding model.
        embeddings = OllamaEmbeddings(model=MODEL_NAME)

        # Create a new Chroma vector store from the documents
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        print("Indexing complete!")
        return jsonify(
            {"status": "success", "message": f"Successfully indexed {len(docs)} files."}
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    API endpoint to ask a question about the indexed code.
    Expects a JSON payload with a 'question' key.
    """
    global vectorstore

    if vectorstore is None:
        return (
            jsonify({"error": "Code not indexed yet. Please index a directory first."}),
            400,
        )

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    question = data["question"]

    try:
        # 1. Retriever
        # This object retrieves the most relevant documents from the vector store.
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        # 2. Prompt Template
        # This template structures the input for the LLM.
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

        # 3. LLM
        llm = OllamaLLM(model=MODEL_NAME, temperature=0)

        # 4. RAG Chain
        # This chain pipes together the retriever, document formatter, prompt, LLM, and output parser.
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 5. Invoke Chain and Get Answer
        print(f"Invoking chain with question: '{question}'")
        answer = rag_chain.invoke(question)

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"An error occurred during question answering: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
