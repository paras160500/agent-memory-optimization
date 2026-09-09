# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm
from common.token_tracker import TokenTracker
from .memory import GraphMemory
from .extractor import GraphExtractor

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class GraphBaseMemoryAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory = GraphMemory()
        self.extractor = GraphExtractor()
        self.token_tracker = TokenTracker(model = "gpt-4o-mini")

    # Chat
    def chat(self , user_message : str) -> str:
        graph_context = self.memory.get_context(user_message)
        prompt = f"""
        You are an AI assistant with graph-based memory.
        Relevant knowledge from memory:
        {graph_context}
        Use the graph facts when they are relevant.
        Current user request:
        {user_message}
        """
        messages = [HumanMessage(content = prompt)]
        self.token_tracker.update_context(messages)

        # Generate response
        response = self.llm.invoke(messages)
        self.token_tracker.record_api_usage(response)

        # Extract knowledge
        full_text = (
            f"User : {user_message}\n"
            f"Assistant : {response.contnet}"
        )
        triples = self.extractor.extract_triples(full_text)

        # Store grpah relationships
        print("\n--- [Graph Memory : Extracted Triples]")
        for subject, relation, obj in triples:
            print(
                f"({subject} , {relation} , {obj})"
            )
            self.memory.add_triple(subject , relation , obj)

        return response.content 

    # Memory
    def get_memory(self):
        return self.memory

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()