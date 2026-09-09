# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from common.llm import get_llm
from common.embeddings import get_embeddings
from common.token_tracker import TokenTracker
from .memory import RetrievalMemory
from .retriever import MemoryRetriever
from langchain_core.messages import HumanMessage

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class RetrievalMemoryAgent:
    def __init__(self , top_k : int = 3):
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.memory = RetrievalMemory()
        self.retriever = MemoryRetriever(top_k)
        self.token_tracker = TokenTracker(model = "gpt-4o-mini")

    # Create Embeddings
    def create_embedding(self , text : str):
        return self.embeddings.embed_query(text)

    # Chat 
    def chat(self , user_message : str) -> str:
        # Convert query to embedding
        query_embedding = self.create_embedding(user_message)
        # Retrieve relevant memories
        retrieved = self.retriever.retrieve(query_embedding , self.memory.get_all_memories())
        # Build memory context
        memory_context = ""
        for score , memory in retrieved:
            memory_context += (f"- {memory.text}\n")
        # Build prompt 
        if memory_context:
            prompt = f"""
                You are an AI assistant with retrieval-based memory.
                Relevant memories from previous conversations:
                {memory_context}
                Use these memories when they are relevant.
                
                User:
                {user_message}
            """
        else:
            prompt = user_message

        # Measure context
        messages = [HumanMessage(content = prompt)]
        self.token_tracker.update_context(messages)

        # Call llm
        response = self.llm.invoke(messages)
        # Record usage
        self.token_tracker.record_api_usage(response)
        # Store current conversation
        memory_text = (
            f"User : {user_message}\n"
            f"Assistant : {response.content}"
        )
        memory_embedding = self.create_embedding(memory_text)
        self.memory.add_memory(memory_text,memory_embedding,"conversation")
        return response.content 

    # Accessor
    def get_memory(self):
        return self.memory
    def get_token_tracker(self):
        return self.token_tracker
    def clear_memory(self):
        self.memory.clear()
        self.token_tracker.reset()