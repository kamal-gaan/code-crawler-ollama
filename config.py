# config.py

import os

# --- Model Configuration ---
# The Ollama model to use for embeddings and generation
OLLAMA_MODEL = "deepseek-coder-v2:16b"
OLLAMA_BASE_URL = "http://localhost:11434"  # Or your Ollama server URL

# --- Vector Store Configuration ---
# Directory to store persistent ChromaDB vector stores
PERSIST_DIRECTORY = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "persistent_storage"
)

# --- Document Processing Configuration ---
# Chunk size and overlap for the text splitter
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File patterns to look for when indexing
# You can make this a list of patterns, e.g., ["**/*.py", "**/*.js"]
TARGET_FILE_GLOB = "**/*.py"

# --- Retriever Configuration ---
# Number of relevant document chunks to retrieve
RETRIEVER_K = 5

# --- Flask Configuration ---
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000

SUMMARIZE_DOCUMENTS = False