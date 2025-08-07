# agent_tools.py

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
        Retrieves the full content of a specific file from the vector store.
        """
        print(f"Tool: Retrieving content for file '{file_path}'")
        # Use metadata filtering to get all chunks for a specific file
        results = self.vectorstore.similarity_search(
            " ",  # Dummy query, the filter is what matters
            filter={"source": file_path, "type": "code"},
            k=100,  # Retrieve a large number of chunks to get the whole file
        )
        if not results:
            return f"Error: Could not find file {file_path} in the collection."

        # Sort chunks by their original position if possible (depends on loader metadata)
        # For now, we assume the order is reasonably preserved.
        return "\n".join([doc.page_content for doc in results])

    def list_functions_to_improve(self, file_content: str) -> str:
        """
        Uses the LLM to identify functions in a file that need docstrings.
        """
        print("Tool: Identifying functions that need docstrings.")
        template = """You are a code analysis expert. Analyze the following Python code and identify all functions that do not have a PEP 257 compliant docstring.
        Respond ONLY with a comma-separated list of the function names. If all functions are documented, return an empty string.

        Example Response: my_function_one,my_function_two

        CODE:
        {code}

        FUNCTION NAMES:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"code": file_content})

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
