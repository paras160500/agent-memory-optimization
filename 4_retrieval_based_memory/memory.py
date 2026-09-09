# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from dataclasses import dataclass

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

@dataclass
class MemoryItem:
    text : str
    embedding : list[float]
    memory_type : str 

class RetrievalMemory:
    def __init__(self):
        self.memories = []

    def add_memory(self, text : str , embedding : list[float] , memory_type : str = "conversation"):
        memory = MemoryItem(
            text = text , embedding = embedding , memory_type = memory_type
        )
        self.memories.append(memory)

    def get_all_memories(self):
        return self.memories

    def count(self):
        return len(self.memories)

    def clear(self):
        self.memories = []