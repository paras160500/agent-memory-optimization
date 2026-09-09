# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

import numpy as np 

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class HierarchicalRetriever:
    def __init__(self , top_k : int = 3):
        self.top_k = self.top_k

    def cosine_similarity(self, query_embedding , memory_embedding):
        query = np.array(query_embedding , dtype=float)
        memory = np.array(memory_embedding , dtype=float)
        denominator = (np.linalg.norm(query) * np.linalg.norm(memory))
        if denominator == 0:
            return 0.0
        return float(np.dot(query , memory) / denominator)

    def retrieve(self , query_embedding , memories):
        scored_memories = []
        for memory in memories:
            score = self.cosine_similarity(query_embedding , memory.embedding)
            scored_memories.append((score , memory))
        scored_memories.sort(key=lambda item : item[0] , reverse=True)
        return scored_memories[ : self.top_k]

    