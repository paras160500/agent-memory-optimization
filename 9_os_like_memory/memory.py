# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from collections import deque

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class OSMemory:

    def __init__(self,ram_size: int = 2):

        # RAM
        self.ram_size = ram_size
        self.active_memory = deque()
        
        # Disk
        self.passive_memory = {}

        # Turn ID
        self.turn_count = 0

        # Statistics
        self.page_faults = 0
        self.pages_out = 0
        self.pages_in = 0

    # Add Turn
    def add_message(self,user_message: str, ai_message: str):
        turn_id = self.turn_count
        turn_data = (
            f"User: {user_message}\n"
            f"AI: {ai_message}"
        )
        # -----------------------------------------------------
        # RAM is full
        if len(self.active_memory) >= self.ram_size:
            # Remove oldest item
            lru_turn_id, lru_turn_data = (
                self.active_memory.popleft()
            )
            # Move to disk
            self.passive_memory[ lru_turn_id] = lru_turn_data

            self.pages_out += 1
            print(
                f"\n--- [OS Memory: "
                f"Paging out Turn "
                f"{lru_turn_id} to disk.] ---"
            )

        # Add new item to RAM
        self.active_memory.append(
            (
                turn_id,
                turn_data
            )
        )
        self.turn_count += 1

    # Search Passive Memory
    def search_passive_memory(self,query: str):
        matches = []
        query_words = [
            word.lower()
            for word in query.split()
            if len(word) > 3
        ]
        for turn_id, data in (self.passive_memory.items()):
            data_lower = data.lower()
            if any(word in data_lower for word in query_words):
                matches.append(
                    (
                        turn_id,
                        data
                    )
                )
        return matches

    # Get Context
    def get_context(self,query: str):
        # RAM context
        active_context = "\n".join(
            data
            for _, data
            in self.active_memory
        )

        # Search disk
        paged_in = (self.search_passive_memory(query))
        paged_in_context = ""
        if paged_in:
            self.page_faults += 1
            print(
                "\n--- [OS Memory: "
                "PAGE FAULT detected!] ---"
            )

            for turn_id, data in paged_in:
                print(
                    f"--- [OS Memory: "
                    f"Paging IN Turn "
                    f"{turn_id} from disk.] ---"
                )
                paged_in_context += (
                    f"\n(Paged in from Turn "
                    f"{turn_id}):\n"
                    f"{data}\n"
                )
                self.pages_in += 1
        return (
            "### Active Memory (RAM)\n"
            f"{active_context}\n\n"
            "### Paged-In Memory (Disk)\n"
            f"{paged_in_context}"
        )

    # Statistics
    def ram_count(self):

        return len(
            self.active_memory
        )

    def disk_count(self):

        return len(
            self.passive_memory
        )

    def get_statistics(self):

        return {
            "ram_items": self.ram_count(),
            "disk_items": self.disk_count(),
            "page_faults": self.page_faults,
            "pages_out": self.pages_out,
            "pages_in": self.pages_in,
        }

    # Clear
    def clear(self):
        self.active_memory.clear()
        self.passive_memory = {}
        self.turn_count = 0
        self.page_faults = 0
        self.pages_out = 0
        self.pages_in = 0