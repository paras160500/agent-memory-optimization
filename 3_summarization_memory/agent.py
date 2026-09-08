# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from common.llm import get_llm
from common.token_tracker import TokenTracker
from .memory import SummarizationMemory
from .summarizer import ConversationSummarizer
from langchain_core.messages import SystemMessage

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class SummarizationMemoryAgent:
    def __init__(self , max_recent_messages : int = 4):
        self.llm = get_llm()
        self.memory = SummarizationMemory(max_recent_messages=max_recent_messages)
        self.summarizer = ConversationSummarizer()
        self.token_tracker = TokenTracker(model = "gpt-4o-mini")

    # Chat
    def chat(self , user_message : str):
        # Add user message
        self.memory.add_user_message(user_message)
        # Check old message need summarization
        old_messages = self.memory.get_old_messages()
        if old_messages:
            # Create new summary
            summary = self.summarizer.summarize(
                existing_summary=self.memory.get_summary(), messages=old_messages
            )
            self.memory.update_summary(summary)
            self.memory.remove_old_messages()

        # Build llm context
        messages = []
        if self.memory.get_summary():
            messages.append(
                SystemMessage(content="Here is the user;s long-term memory:\n\n" + self.memory.get_summary())
            )
        messages.extend(self.memory.get_recent_messages())

        # Measure context
        self.token_tracker.update_context(messages)
        response = self.llm.invoke(messages)
        self.token_tracker.record_api_usage(response)
        self.memory.add_ai_message(response.content)
        return response.content

    def get_memory(self):
        return self.memory

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()