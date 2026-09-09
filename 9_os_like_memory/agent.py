# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm
from common.token_tracker import TokenTracker
from .memory import OSMemory

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class OSMemoryAgent:

    def __init__(self,ram_size: int = 2):
        self.llm = get_llm()
        self.memory = OSMemory(ram_size=ram_size)
        self.token_tracker = TokenTracker(model="gpt-4o-mini")

    # Chat
    def chat( self, user_message: str) -> str:
        # STEP 1
        # Get memory context
        memory_context = (
            self.memory.get_context(
                user_message
            )
        )

        # STEP 2
        # Build prompt
        prompt = f"""
            You are an AI assistant with OS-like memory management.

            Your memory system has:

            1. Active Memory (RAM)
            - Fast
            - Limited capacity

            2. Passive Memory (Disk)
            - Larger capacity
            - Used when information is paged out

            Relevant memory:

            {memory_context}

            Use the available memory when answering.

            Current request:

            {user_message}
        """

        messages = [HumanMessage(content=prompt)]

        # STEP 3
        # Token tracking
        self.token_tracker.update_context(messages)

        # STEP 4
        # Generate response
        response = self.llm.invoke(messages)
        self.token_tracker.record_api_usage( response)

        # STEP 5
        # Add current turn to memory
        self.memory.add_message(user_message,response.content)
        return response.content

    # Memory
    def get_memory(self):
        return self.memory

    # Tracker
    def get_token_tracker(self):
        return self.token_tracker

    # Clear
    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()