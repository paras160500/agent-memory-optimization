# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm
from common.token_tracker import TokenTracker
from .memory import CompressionMemory
from .compressor import MemoryCompressor

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class CompressionMemoryAgent:

    def __init__(self):
        self.llm = get_llm()
        self.memory = CompressionMemory()
        self.compressor = MemoryCompressor()
        self.token_tracker = TokenTracker(model="gpt-4o-mini")

    # Chat
    def chat(self,user_message: str) -> str:
        # 1 : REtrieve compressed memory
        memory_context = (self.memory.get_context())
        # 2 : Build prompt
        prompt = f"""
            You are an AI assistant with compressed memory.

            Previous compressed facts:

            {memory_context}

            Use these facts when relevant.

            Current request:

            {user_message}
        """
        messages = [HumanMessage(content=prompt)]
        self.token_tracker.update_context(messages)

        response = self.llm.invoke(messages)
        self.token_tracker.record_api_usage(response)
        compressed_fact = (self.compressor.compress(user_message,response.content))

        print("\n--- [Compression Memory: New compressed fact] ---")
        print(compressed_fact)
        self.memory.add_fact(compressed_fact)
        return response.content

    def get_memory(self):
        return self.memory

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()