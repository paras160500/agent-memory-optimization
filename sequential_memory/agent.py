# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from common.llm import get_llm
from .memory import SequentialMemory
from common.token_tracker import TokenTracker

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SequentialMemoryAgent:

    def __init__(self):
        self.llm = get_llm()
        self.memory = SequentialMemory()
        self.token_tracker = TokenTracker(model="gpt-4o-mini")

    # Chat
    def chat(self , user_message : str) -> str:
        # Add user message to memory
        self.memory.add_user_message(user_message)
        # Get complete conversation history
        messages = self.memory.get_messages()
        # MEasure context before sending
        self.token_tracker.update_context(messages)
        response = self.llm.invoke(messages)
        # Record actual OPENAI Usage
        self.token_tracker.record_api_usage(response)
        # Store AI Response
        self.memory.add_ai_message(response.content)
        return response.content 

    def get_memory(self):
        return self.memory.get_messages()

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()