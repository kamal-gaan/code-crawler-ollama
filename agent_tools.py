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

        # --- FINAL, POLISHED PROMPT ---
        # This version clarifies the instructions and simplifies the examples to reduce confusion.
        template = """You are a code analysis tool. Your only job is to find Python functions without docstrings in the provided code.
        You must follow these rules strictly:
        1.  Analyze the user's code provided after "---CODE---".
        2.  Respond ONLY with a comma-separated list of the function names that are missing docstrings.
        3.  DO NOT include any functions from the examples.
        4.  DO NOT add any conversational text, explanations, or bullet points.
        5.  If no functions are missing docstrings, you MUST return the exact word "NONE".

        ---EXAMPLE 1---
        [CODE]
        def func_with_docstring():
            '''I have a docstring.'''
            pass
        def func_without_docstring(x):
            return x + 1
        [RESPONSE]
        func_without_docstring

        ---EXAMPLE 2---
        [CODE]
        def another_documented_func():
            '''Also documented.'''
            pass
        [RESPONSE]
        NONE

        ---CODE---
        {code}

        [RESPONSE]"""
        # --- END OF FINAL PROMPT ---

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"code": file_content})

        response = response.strip()
        if "NONE" in response or not response:
            return ""

        # Extra safeguard: remove any example function names if the LLM still includes them
        example_funcs = ["func_without_docstring", "another_documented_func"]
        found_funcs = [
            func.strip()
            for func in response.split(",")
            if func.strip() and func.strip() not in example_funcs
        ]

        return ",".join(found_funcs)

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

    def save_modified_code(self, file_path: str, modified_code: str) -> str:
        """
        Saves the modified code back to the original file path.
        """
        print(f"Tool: Saving improved code to '{file_path}'")
        try:
            # It's a good practice to make a backup of the original file
            backup_path = file_path + ".bak"
            os.rename(file_path, backup_path)
            print(f"Tool: Original file backed up at '{backup_path}'")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_code)
            
            return f"Successfully saved changes to {os.path.basename(file_path)}."
        except Exception as e:
            error_message = f"Error saving file {file_path}: {e}"
            print(f"Tool: {error_message}")
            return error_message