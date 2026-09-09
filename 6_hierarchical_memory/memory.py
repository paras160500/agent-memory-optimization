# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from dataclasses import dataclass

@dataclass
class MemoryItem:
    text : str 
    embedding : list[float]
    memory_type : str

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class HierarchicalMemory:
    def __init__(self , window_size : int = 4 , top_k :int = 3):
        # Working memory
        self.window_size = window_size
        self.working_memory = []

        # Long term memory
        self.long_term_memory = []
        self.top_k = top_k
        self.promotion_keywords = [
            "remember" , "preference" , "prefer" , "always" ,  "never" , "important" , "rule"
        ]

    # Working memory
    def add_to_working_memory(self , user_message : str , ai_message : str):
        self.working_memory.append(
            {"role" : "user" , "content" : user_message}
        )
        self.working_memory.append(
            {"role" : "assistant" , "content" : ai_message}
        )
        if len(self.working_memory) > self.window_size:
            self.working_memory = self.working_memory[-self.window_size : ]

    # Long-term memory
    def add_to_long_term_memory(self , text : str , embedding : list[float] , memory_type : str = "important_fact"):
        memory = MemoryItem(text , embedding , memory_type)
        self.long_term_memory.append(memory)

    # Promotion Decision
    def should_promote(self , user_message : str) -> bool:
        message = user_message.lower()
        return any(
            keyword in message for keyword in self.promotion_keywords
        )

    # Get working memory
    def get_working_memory(self):
        return self.working_memory

    # Get long-term memory
    def get_long_term_memory(self):
        return self.long_term_memory

    # counts
    def working_count(self):
        return len(self.working_memory)

    def long_term_count(self):
        return len(self.long_term_memory)

    def clear(self):
        self.working_memory = []
        self.long_term_memory = []
