# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from dataclasses import dataclass

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

@dataclass 
class MemorySlot:
    key : list[float]
    value : str 

class ExternalMemory:
    def __init__(self):
        self.slots = []
    def add(self , key : list[float] , value : str):
        self.slots.append(MemorySlot(key=key , value=value))
    def get_slots(self):
        return self.slots
    def count(self):
        return len(self.slots)
    def clear(self):
        self.slots = [] 