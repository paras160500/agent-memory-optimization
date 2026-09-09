# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

import ast 
from langchain_core.messages import HumanMessage
from common.llm import get_llm

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class GraphExtractor:
    def __init__(self):
        self.llm = get_llm()

    def extract_triples(self , text : str):
        prompt = f"""
        You are a knowledge graph extraction engine.
        Extract factual relationships from the text.
        Return ONLY a python list of tuples.
        Each tuples must have:
        (subject , relation , object)
        Example:
        [
            ("Paras" , "works_on" , "AI Memory Lab"),
            ("AI Memory Lab" , "uses" , "Python")
        ]
        if there are no useful relationships, return:
        []
        Text:
        {text}
        """
        response = self.llm.invoke(
            [HumanMessage(content = prompt)]
        )
        try:
            triples = ast.literal_eval(response.content)
            if not isinstance(triples , list):
                return []
            valid_triples = []
            for triple in triples:
                if(isinstance(triple , tuple) and len(triple) == 3):
                    valid_triples.append(triple)
            return valid_triples
        except Exception:
            return []