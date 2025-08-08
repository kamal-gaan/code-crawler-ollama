# agent_tools.py
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma


class CodeAgentTools:
    def __init__(self, llm, embeddings, collection_name, persist_directory):
        self.llm = llm
        self.vectorstore = Chroma(
            persist_directory=f"{persist_directory}/{collection_name}",
            embedding_function=embeddings,
        )

    def get_full_file_content(self, file_path: str) -> str:
        """
        Retrieves the full content of a specific file from the vector store using a direct metadata query.
        This version normalizes the file path to match the OS-specific format.
        """
        print(f"Tool: Retrieving content for file '{file_path}'")

        try:
            # --- THIS IS THE DEFINITIVE FIX ---
            # Normalize the path to match the format likely stored by the loader on your OS (e.g., using '\' on Windows)
            normalized_path = os.path.normpath(file_path)
            print(f"Tool: Normalized path to '{normalized_path}' for querying.")
            # --- END OF FIX ---

            results = self.vectorstore.get(
                where={
                    "$and": [
                        {
                            "source": {"$eq": normalized_path}
                        },  # Use the normalized path in the query
                        {"type": {"$eq": "code"}},
                    ]
                },
                limit=1000,
            )

            documents = results.get("documents", [])

            if not documents:
                # Add extra debug info if it fails again
                print(
                    f"DEBUG: File not found. Run debug_db.py to see all available file paths in the 'temp' collection."
                )
                return (
                    f"Error: Could not find file {normalized_path} in the collection."
                )

            return "\n".join(documents)

        except Exception as e:
            print(f"An error occurred while querying the vector store: {e}")
            return f"Error: An exception occurred while trying to retrieve {file_path}."

    def list_functions_to_improve(self, file_content: str) -> str:
        """
        Uses the LLM to identify functions in a file that need docstrings with a very strict prompt.
        """
        print("Tool: Identifying functions that need docstrings.")
        
        # --- NEW, STRICTER PROMPT ---
        template = """You are a highly disciplined code analysis tool. Your sole purpose is to identify Python functions that do not have a docstring.
        Analyze the following Python code.
        Respond ONLY with a comma-separated list of the function names.
        - DO NOT add any explanation.
        - DO NOT use bullet points.
        - DO NOT say "Here are the functions".
        - If all functions are documented, return the exact string "NONE".

        Example 1:
        CODE:
        def func_a():
            '''My docstring.'''
            pass
        def func_b(x):
            return x + 1
        RESPONSE:
        func_b

        Example 2:
        CODE:
        def func_c():
            '''My docstring.'''
            pass
        RESPONSE:
        NONE

        Here is the code to analyze:
        CODE:
        {code}

        RESPONSE:"""
        # --- END OF NEW PROMPT ---

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"code": file_content})
        
        # Clean up the response just in case
        response = response.strip()
        if response == "NONE":
            return ""
        return response

    def add_docstring_to_function(self, function_code: str) -> str:
        """
        Uses the LLM to add a docstring to a single function's code.
        """
        print(f"Tool: Adding docstring to function:\n{function_code[:80]}...")
        template = """You are an expert Python programmer. Your task is to add a comprehensive and accurate PEP 257 compliant docstring to the following Python function.
        Return ONLY the full, rewritten function code with the new docstring. Do NOT change any of the existing logic.

        INPUT FUNCTION:
        {function}

        REWRITTEN FUNCTION:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"function": function_code})
        # Clean up potential markdown code fences from the LLM response
        return response.strip().replace("```python", "").replace("```", "").strip()
