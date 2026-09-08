# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage, AIMessage

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SummarizationMemory:

    def __init__(self , max_recent_messages : int = 4):
        self.max_recent_messages = max_recent_messages
        self.summary = ""
        self.messages = []

    # Add messages
    def add_user_message(self , message : str):
        self.messages.append(HumanMessage(content = message))
    def add_ai_message(self , message : str):
        self.messages.append(AIMessage(content=message))

    # Summary
    def update_summary(self , summary : str):
        self.summary = summary
    def get_summary(self):
        return self.summary

    # Recent messages
    def get_recent_messages(self):
        return self.messages[-self.max_recent_messages : ]

    # Message for summarization
    def get_old_messages(self):
        if len(self.messages) <= self.max_recent_messages:
            return []
        return self.messages[ : -self.max_recent_messages]

    # Remove summarized messages
    def remove_old_messages(self):
        if len(self.messages) > self.max_recent_messages:
            self.messages = self.messages[-self.max_recent_messages : ]

    # Clear
    def clear(self):
        self.summary = ""
        self.messages = []