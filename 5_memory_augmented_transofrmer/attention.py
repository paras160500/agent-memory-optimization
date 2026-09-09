# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

import numpy as np 

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class MemoryAttention:
    def __init__(self , top_k : int = 3):
        self.top_k = top_k

    # Normalize vector
    def normalize(self , vector):
        vector = np.array(vector , dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm 

    # Attention score
    def attention_score(self , query , key):
        query = self.normalize(query)
        key = self.normalize(key)
        return float(np.dot(query , key))

    # Attend to memory
    def attend(self , query_embedding , memory_slots):
        if not memory_slots:
            return []
        scores = []
        for slot in memory_slots:
            score = self.attention_score(query_embedding , slot.key)
            scores.append((score , slot))

        scores.sort(key = lambda item : item[0] , reverse=True)
        return scores[ : self.top_k]

    