# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class MemoryCompressor:

    def __init__(self):
        self.llm = get_llm()

    def compress(self , user_message : str , ai_message : str):
        conversation = f"""
            User:
            {user_message}
            Assistant:
            {ai_message}
        """

        prompt = f"""
            You are a memory compression engine.
            you job is to convert the following conversation into the
            samllest useful factual memory.

            Remove:

            - Greetings
            - Conversational filler
            - Repeated information
            - Explanations that are no longer needed
            - Unnecessary wording

            Preserve:

            - Important facts
            - Names
            - Preferences
            - Decisions
            - Dates
            - Goals
            - Important entities
            - Important relationships

            Return ONLY the compressed factual statement.

            Conversation:

            {conversation}
        """

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        return response.content.strip()

      