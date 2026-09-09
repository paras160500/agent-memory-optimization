# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_core.messages import HumanMessage
from common.llm import get_llm
from common.embeddings import get_embeddings
from common.token_tracker import TokenTracker
from .memory import HierarchicalMemory
from .retriever import HierarchicalRetriever

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class HierarchicalMemoryAgent:
    def __init__(self , window_size : int = 4 , top_k : int = 3):
        self.llm = get_llm()
        self.embedding = get_embeddings()
        self.memory = HierarchicalMemory(window_size , top_k)
        self.retriever = HierarchicalRetriever(top_k)
        self.token_tracker = TokenTracker(model="gpt-4o-mini")

    # Embedding
    def create_embedding(self,text : str):
        return self.embedding.embed_query(text)

    # Chat
    def chat(self , user_message : str) -> str:
        # Search in longterm memory
        query_embedding = self.create_embedding(user_message)
        retrieved = self.retriever.retrieve(query_embedding , self.memory.get_long_term_memory())
        long_term_context = ""

        for score , memory in retrieved:
            long_term_context += (f"- {memory.text}\n")

        # Getting working memory
        working_context = ""
        for message in self.memory.get_working_memory():
            working_context += (
                f"{message['role'].capitalize()}"
                f"{message['content']}\n"
            )

        # Build context
        prompt = f"""
            You are an AI assistant with hierarchical memory.
            You have two memoy levels.

            LEVEL 1 - WORKING MEMORY
            Recent conversation:
            {working_context}

            LEVEL 2 - LONG-TERM MEMORY
            Important information retrieved from previous conversations:
            {long_term_context if long_term_context else "No relevant long-term memory found."}

            Use long-term memory only when relevant.
            Use working memory for recent conversational context.
            Current user request:
            {user_message}
        """

        messages = [
            HumanMessage(content = prompt)
        ]
        self.token_tracker.update_context(messages)
        response = self.llm.invoke(messages)
        self.token_tracker.record_api_usage(response)
        self.memory.add_to_working_memory(user_message , response.content)

        if self.memory.should_promote(user_message):
            print("\n--- [Hierarchical Memory : Promoting information to long-term memory.]")
            memory_text = (
                f"User : {user_message}\n"
                f"Assistant : {response.content}"
            )
            memory_embedding = self.create_embedding(memory_text)
            self.memory.add_to_long_term_memory(memory_text , memory_embedding)

        return response.content 

    # Memory
    def get_memory(self):
        return self.memory

    def get_token_tracker(self):
        return self.token_tracker

    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()