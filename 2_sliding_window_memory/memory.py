# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage, AIMessage

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SlidingWindowMemory:

    def __init__(self , window_size : int = 4):
        self.window_size = window_size
        self.messages = [] 

    def _trim_memory(self):
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size : ]

    def add_user_message(self , message : str):
        self.messages.append(HumanMessage(content=message))
        self._trim_memory()

    def add_ai_message(self, message : str):
        self.messages.append(AIMessage(content = message))
        self._trim_memory()

    def get_messages(self):
        return self.messages

    def get_message_count(self):
        return len(self.messages)

    def clear(self):
        self.messages = []