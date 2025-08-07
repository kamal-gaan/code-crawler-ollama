# app.py

import os
from flask import Flask, request, jsonify, render_template

# --- Our Custom Modules ---
from rag_service import RAGService
from agent_tools import CodeAgentTools
from code_agent import CodeImprovementAgent

# --- LangChain Core Imports ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# --- Flask App Setup ---
app = Flask(__name__)

# Load configuration from config.py
# Make sure you have a config.py file in the same directory
try:
    import config

    app.config.from_object(config)
except ImportError:
    print("Warning: config.py not found. Using default settings.")
    # Define default fallbacks here if needed
    app.config.setdefault("OLLAMA_MODEL", "codellama")
    app.config.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    app.config.setdefault("PERSIST_DIRECTORY", "persistent_storage")
    app.config.setdefault("HOST", "0.0.0.0")
    app.config.setdefault("PORT", 5000)
    app.config.setdefault("DEBUG_MODE", True)

# --- Flask Routes ---


@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/api/collections", methods=["POST"])
def index_collection():
    """
    API endpoint to index a new codebase (a 'collection').
    Now with a check to prevent re-indexing an existing collection.
    """
    data = request.get_json()
    collection_name = data.get("collection_name")
    code_path = data.get("path")

    if not all([collection_name, code_path]):
        return jsonify({"error": "Both 'collection_name' and 'path' are required"}), 400

    # --- NEW: Check if the collection directory already exists ---
    collection_persist_path = os.path.join(
        app.config["PERSIST_DIRECTORY"], collection_name
    )
    if os.path.exists(collection_persist_path):
        return (
            jsonify(
                {
                    "status": "skipped",
                    "message": f"Collection '{collection_name}' already exists. No need to re-index.",
                }
            ),
            200,
        )  # Return 200 OK instead of creating a new resource
    # --- END OF NEW CHECK ---

    if not os.path.isdir(code_path):
        return jsonify({"error": f"Invalid path specified: {code_path}"}), 400

    try:
        service = RAGService(collection_name=collection_name)
        num_docs = service.index_codebase(code_path)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Successfully indexed {num_docs} files into collection '{collection_name}'.",
                }
            ),
            201,
        )

    except Exception as e:
        print(f"An error occurred during indexing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/collections/<collection_name>/ask", methods=["POST"])
def ask_question(collection_name: str):
    """
    API endpoint to ask a question about a specific indexed collection.
    """
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "A question is required"}), 400

    try:
        service = RAGService(collection_name=collection_name)
        if not service.vectorstore:
            return (
                jsonify(
                    {
                        "error": f"Collection '{collection_name}' not found or not indexed yet."
                    }
                ),
                404,
            )

        answer = service.ask_question(question)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"An error occurred during question answering: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/collections/<collection_name>/improve", methods=["POST"])
def improve_code(collection_name: str):
    """
    API endpoint to run the code improvement agent on a specific file.
    """
    data = request.get_json()
    file_path = data.get("file_path")
    task = data.get("task", "add docstrings")

    if not file_path:
        return jsonify({"error": "file_path is required"}), 400

    print(f"Starting agent for collection '{collection_name}' on file '{file_path}'")
    try:
        llm = OllamaLLM(
            model=app.config["OLLAMA_MODEL"], base_url=app.config["OLLAMA_BASE_URL"]
        )
        embeddings = OllamaEmbeddings(
            model=app.config["OLLAMA_MODEL"], base_url=app.config["OLLAMA_BASE_URL"]
        )

        tools = CodeAgentTools(
            llm, embeddings, collection_name, app.config["PERSIST_DIRECTORY"]
        )
        agent = CodeImprovementAgent(tools)

        result_state = agent.run(collection_name, file_path, task)

        if result_state.get("error"):
            return jsonify({"error": result_state["error"]}), 500

        return jsonify(
            {
                "message": "Agent run complete.",
                "modified_code": result_state.get(
                    "final_code", "No changes were made."
                ),
            }
        )

    except Exception as e:
        print(f"An error occurred during agent run: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create the main persistence directory if it doesn't exist
    os.makedirs(app.config["PERSIST_DIRECTORY"], exist_ok=True)
    app.run(
        host=app.config["HOST"], port=app.config["PORT"], debug=app.config["DEBUG_MODE"]
    )
