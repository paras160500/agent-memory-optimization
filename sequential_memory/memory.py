# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage, AIMessage

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SequentialMemory:

    def __init__(self):
        self.messages = []

    # Add User Message 
    def add_user_message(self , message : str):
        self.messages.append(
            HumanMessage(content = message)
        )

    # Add AI Message
    def add_ai_message(self , message : str):
        self.messages.append(
            AIMessage(content = message)
        )

    # Get Messages
    def get_messages(self):
        return self.messages

    # Clear messages
    def clear(self):
        self.messages = []