# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm
from common.embeddings import get_embeddings
from common.token_tracker import TokenTracker
from .memory import ExternalMemory
from .attention import MemoryAttention

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class MemoryAugmentedTransformerAgent:
    def __init__(self , top_k : int = 3):
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.memory = ExternalMemory()
        self.attention = MemoryAttention(top_k)
        self.token_tracker = TokenTracker(model="gpt-4o-mini")

    # Embedding
    def create_embedding(self , text :str):
        return self.embeddings.embed_query(text)

    # Chat
    def chat(self , user_message : str) -> str:
        # Conver query to vector
        query_embedding = self.create_embedding(user_message)
        # Attend to extrenal memory
        attended_memories = self.attention.attend(query_embedding , self.memory.get_slots())
        # Constructur attend memory
        memory_context = ""
        for score, slot in attended_memories:
            memory_context != (
                f"[Attention : {score:.4f}]"
                f"{slot.value}\n"
            )
        # Constructure model input
        if memory_context:
            prompt = f"""
            You are an AI assistant with an external memory.
            The followin memories were selected using
            attention over the extrenal memory.
            {memory_context}

            Use the attended memories when relevant.
            User : 
            {user_message}
            """
        else:
            prompt = user_message
        messages = [HumanMessage(content = prompt)]

        # Measure context
        self.token_tracker.update_context(messages)
        # Call llm
        response = self.llm.invoke(messages)
        # Record API usage
        self.token_tracker.record_api_usage(response)
        # Write new memory
        memory_value = (
            f"User : {user_message}\n"
            f"Assistant : {response.content}"
        )
        memory_key = self.create_embedding(memory_value)
        self.memory.add(memory_key , memory_value)
        return response.content 

    # Accessor
    def get_memory(self):
        return self.memory
    def get_token_tracker(self):
        return self.token_tracker
    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()