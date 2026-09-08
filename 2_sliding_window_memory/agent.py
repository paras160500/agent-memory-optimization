# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from common.llm import get_llm
from common.token_tracker import TokenTracker
from .memory import SlidingWindowMemory

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SlidingWindowMemoryAgnet:

    def __init__(self , window_size : int = 4):
        self.llm = get_llm()
        self.memory = SlidingWindowMemory(window_size=window_size)
        self.token_tracker = TokenTracker(model = "gpt-4o-mini")

    def chat(self , user_message : str) -> str:
        # Add user message 
        self.memory.add_user_message(user_message)
        # Get only messages inside window
        messages = self.memory.get_messages()
        # Measure current context
        self.token_tracker.update_context(messages)
        # Send only the window to the llm
        response = self.llm.invoke(messages)
        # Record the API usage
        self.token_tracker.record_api_usage(response)
        # Store AI response
        self.memory.add_ai_message(response.content)
        return response.content

    def get_memory(self):
        return self.memory.get_messages()

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()