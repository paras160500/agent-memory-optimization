# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import SystemMessage
from common.llm import get_llm

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class ConversationSummarizer:
    def __init__(self):
        self.llm = get_llm()

    def summarize(self , existing_summary : str , messages) -> str:
        conversation_text = ""
        for message in messages:
            role = message.type
            conversation_text += f"{role} : {message.content}\n"

        prompt = f"""
            You are maintaining long-term meomry for an AI agent.
            
            Existing memory summary:
            {existing_summary if existing_summary else "No existing summary."}

            New Conversation:
            {conversation_text}

            Create a concise but infomation-rick memory summary.

            Preserve important information such as:
            - User identity
            - User preferences
            - Important facts
            - Projects
            - Goals
            - Decisions
            - Important context

            Do not include unnecessary conversational details.
            Return only the updated summary.
        """

        response = self.llm.invoke(
            [SystemMessage(content = prompt)]
        )
        return response.content