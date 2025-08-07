# Local Codebase Q\&A and Refactoring Agent

This project provides a powerful, locally-run toolkit for developers to interact with their codebases using Large Language Models (LLMs). It features a web-based interface built with Flask that allows you to index a local code repository and then perform two main tasks:

1. **Conversational Q\&A:** Ask questions about your code in natural language (e.g., "What does the `RAGService` class do?").
2. **AI-Powered Code Improvement:** Deploy an autonomous agent (built with LangGraph) to perform tasks like automatically adding docstrings to functions in a specified file.

The entire system runs locally using **Ollama**, ensuring your code remains private and secure.

*(**Note:** You should replace this placeholder with an actual screenshot of your application's web interface\!)*

## ✨ Features

* **Local LLM Integration:** Powered by [Ollama](https://ollama.com/) to run models like `codellama` locally on your machine.
* **Intelligent Codebase Indexing:**
  * Recursively scans a directory for source code files.
  * Automatically respects `.gitignore` rules to exclude irrelevant files.
  * **Optional Hierarchical Indexing:** Can generate high-level summaries for each file to provide better context to the LLM.
* **Persistent Vector Storage:** Uses [ChromaDB](https://www.trychroma.com/) to save the indexed codebase, so you only need to perform the time-consuming indexing process once.
* **Conversational Q\&A:** Chat with your code to understand its structure, functionality, and logic.
* **Autonomous Code Agent:**
  * Built with the powerful [LangGraph](https://langchain-ai.github.io/langgraph/) library.
  * Can perform automated tasks, with the initial tool being "add docstrings to a file".
  * The agent is designed to be extensible with new tools and capabilities.
* **Simple Web Interface:** A clean UI built with [Flask](https://flask.palletsprojects.com/) to easily manage indexing and interaction.
* **Configurable:** Easily change the LLM model, summarization settings, and other parameters via a `config.py` file.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **LLM Serving:** Ollama
* **AI Orchestration:** LangChain, LangGraph
* **Vector Database:** ChromaDB
* **Package Management:** uv
* **File Parsing:** `unstructured`, `pathspec`

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

1. **Python:** Ensure you have Python 3.12+ installed.
2. **Ollama:** You must have Ollama installed and running. You can download it from [ollama.com](https://ollama.com/).
3. **LLM Model:** After installing Ollama, pull the `codellama` model by running this command in your terminal:

    ```bash
    ollama pull codellama
    ```

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kamalpandi/code-crawler-ollama.git
    cd code-crawler-ollama
    ```

2. **Create a virtual environment using `uv`:**

    ```bash
    uv venv
    ```

3. **Activate the virtual environment:**

      * On Windows (PowerShell):

        ```powershell
        .venv\Scripts\Activate.ps1
        ```

      * On macOS/Linux:

        ```bash
        source .venv/bin/activate
        ```

4. **Install all required dependencies using `uv`:**
    This command reads the `pyproject.toml` file and installs everything needed.

    ```bash
    uv pip sync
    ```

## ⚙️ Configuration

The application's behavior can be customized in the `config.py` file.

* `OLLAMA_MODEL`: The name of the model to use from Ollama (e.g., `"codellama"`).
* `SUMMARIZE_DOCUMENTS`: A crucial flag for indexing.
  * `True`: The application will generate a summary for every document. This provides much better context for the LLM but is **very slow** (can take hours).
  * `False`: Skips the summarization step for **very fast** indexing (usually 1-2 minutes). Recommended for most cases.

## Usage

1. **Start the Flask application:**

    ```bash
    python app.py
    ```

    Wait for the server to start, then open your web browser to `http://127.0.0.1:5000`.

2. **Indexing a Codebase (Section 1):**

      * Enter the absolute path to the code repository you want to analyze in the first input box.
      * Click "Index Code".
      * Wait for the process to complete. You will see progress in the terminal and a success message on the webpage. This creates a persistent database in the `persistent_storage` directory.

3. **Asking Questions (Section 2):**

      * Once a codebase is indexed, you can ask questions about it in the second input box.
      * Click "Ask Question" to get an answer from the LLM based on the indexed context.

4. **Improving Code (Section 3):**

      * Enter the path to a specific file *relative to the indexed directory* (e.g., `app.py` or `src/utils.py`).
      * Click "Run Improvement Agent".
      * The agent will process the file (e.g., add docstrings) and display the modified code on the page for you to review.

## 📈 Future Improvements

This project has a solid foundation that can be extended with many new features:

* **More Agent Tools:** Create new tools for the agent, such as:
  * Writing unit tests for a function.
  * Refactoring code to adhere to PEP 8.
  * Converting a function from one language/framework to another.
* **Asynchronous Indexing:** Use a task queue like Celery and Redis to run the slow indexing process in the background without blocking the web interface.
* **Side-by-Side Diff View:** Instead of just showing the new code, display a "diff" view (like in GitHub) to clearly highlight the agent's changes.
* **Support for More File Types:** Extend the loader to handle other document types like Jupyter Notebooks or technical documentation.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
