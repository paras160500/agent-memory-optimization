# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class CompressionMemory:
    def __init__(self):
        self.compressed_facts = []

    # Add Fact
    def add_fact(self , compressed_fact : str):
        self.compressed_facts.append(compressed_fact)

    # Get Fatcs
    def get_facts(self):
        return self.compressed_facts

    # Get context
    def get_context(self):
        if not self.compressed_facts:
            return (
                "No compressed facts in memory"
            )
        return (
            "### Compressed Factual Memory:\n"
            +"\n".join(f"-{fact}" for fact in self.compressed_facts)
        )

    # Count
    def count(self):
        return len(self.compressed_facts)

    # Clear
    def clear(self):
        self.compressed_facts = []