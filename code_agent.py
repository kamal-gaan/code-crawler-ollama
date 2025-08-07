# code_agent.py

import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agent_tools import CodeAgentTools


class AgentState(TypedDict):
    """Defines the state of our agent. This is passed between nodes."""

    collection_name: str
    file_path: str
    task_description: str
    full_file_content: str
    functions_to_improve: List[str]
    modified_functions: dict  # Mapping of function_name -> modified_code
    final_code: str
    error: str


class CodeImprovementAgent:
    def __init__(self, tools: CodeAgentTools):
        self.tools = tools
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        # Define the nodes of the graph
        graph.add_node("retrieve_file", self.retrieve_file_node)
        graph.add_node("find_functions", self.find_functions_node)
        graph.add_node("improve_function", self.improve_function_node)
        graph.add_node("assemble_code", self.assemble_code_node)
        graph.add_node("handle_error", self.handle_error_node)

        # Define the edges (the flow of control)
        graph.set_entry_point("retrieve_file")
        graph.add_edge("retrieve_file", "find_functions")
        graph.add_conditional_edges(
            "find_functions",
            self.decide_to_improve_or_finish,
            {"continue": "improve_function", "finish": END, "error": "handle_error"},
        )
        graph.add_conditional_edges(
            "improve_function",
            self.decide_to_continue_improving,
            {"continue": "improve_function", "finish": "assemble_code"},
        )
        graph.add_edge("assemble_code", END)
        graph.add_edge("handle_error", END)

        return graph.compile()

    # --- Node Implementations ---

    def retrieve_file_node(self, state: AgentState):
        print("Node: Retrieving file content...")
        content = self.tools.get_full_file_content(state["file_path"])
        if content.startswith("Error:"):
            state["error"] = content
        state["full_file_content"] = content
        return state

    def find_functions_node(self, state: AgentState):
        print("Node: Finding functions to improve...")
        if state.get("error"):
            return state

        function_names_str = self.tools.list_functions_to_improve(
            state["full_file_content"]
        )
        if not function_names_str.strip():
            print("No functions to improve.")
            state["functions_to_improve"] = []
        else:
            state["functions_to_improve"] = [
                name.strip() for name in function_names_str.split(",")
            ]
        state["modified_functions"] = {}
        return state

    def improve_function_node(self, state: AgentState):
        print("Node: Improving a function...")
        if state.get("error"):
            return state

        # Process one function at a time from the list
        func_name = state["functions_to_improve"].pop(0)

        # Simple regex to extract a function's code block
        func_regex = re.compile(
            rf"def {func_name}\(.*\):(?:\n(?!\s*def\s).|.)*", re.DOTALL
        )
        match = func_regex.search(state["full_file_content"])

        if not match:
            print(f"Warning: Could not find code for function '{func_name}'. Skipping.")
            return state

        original_func_code = match.group(0)
        modified_func_code = self.tools.add_docstring_to_function(original_func_code)
        state["modified_functions"][func_name] = (
            original_func_code,
            modified_func_code,
        )

        return state

    def assemble_code_node(self, state: AgentState):
        print("Node: Assembling final code...")
        if state.get("error"):
            return state

        final_code = state["full_file_content"]
        for func_name, (original, modified) in state["modified_functions"].items():
            final_code = final_code.replace(original, modified, 1)
        state["final_code"] = final_code
        return state

    def handle_error_node(self, state: AgentState):
        print(f"Node: Handling error - {state['error']}")
        return state

    # --- Conditional Edge Logic ---

    def decide_to_improve_or_finish(self, state: AgentState):
        if state.get("error"):
            return "error"
        if not state.get("functions_to_improve"):
            return "finish"
        return "continue"

    def decide_to_continue_improving(self, state: AgentState):
        if state.get("error"):
            return "error"
        if not state.get("functions_to_improve"):
            return "finish"
        return "continue"

    # --- Agent's Public run Method ---

    def run(self, collection_name: str, file_path: str, task_description: str):
        initial_state = AgentState(
            collection_name=collection_name,
            file_path=file_path,
            task_description=task_description,
            full_file_content="",
            functions_to_improve=[],
            modified_functions={},
            final_code="",
            error=None,
        )
        final_state = self.graph.invoke(initial_state)
        return final_state
