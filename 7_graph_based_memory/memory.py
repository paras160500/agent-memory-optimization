# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

import networkx as nx

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

class GraphMemory:
    def __init__(self):
        self.graph = nx.DiGraph()

    # Add triple
    def add_triple(self, subject : str , relation : str , obj : str):
        self.graph.add_edge(subject.strip() , obj.strip() , relation = relation.strip())

    # Get related facts
    def get_related_facts(self , entity : str):
        facts = []
        for u,v,data in self.graph.out_edges(entity , data = True):
            facts.append(f"{u} --[{data['relation']}]--> {v}")

        # Incoming relation
        for u,v,data in self.graph.in_edges(entity , data = True):
            facts.append(f"{u} --[{data['relation']}]--> {v}")

        return facts 

    # Find entities
    def find_entities(self,query : str):
        query_words = (query.replace("?","").replace("," , "").split())
        entities = []

        for word in query_words:
            cleaned_word = word.strip("'\".")
            if cleaned_word in self.graph.nodes:
                entities.append(cleaned_word)
            elif cleaned_word.capitalize() in self.graph.nodes:
                entities.append(cleaned_word.capitalize())

        return list(set(entities))

    # Context
    def get_context(self , query : str):
        if not self.graph.nodes:
            return "Knowledge graph is empty."
        entities = self.find_entities(query)
        if not entities:
            return ("No relevant entities were found in the knowledge graph")
        facts = []
        for entity in entities:
            facts.extend(self.get_related_facts(entity))
        facts = sorted(set(facts))
        return (
            "### Facts Retrieved"
            "from knowledge Graph:\n"
            + "\n".join(facts)
        )

    # Counts
    def node_count(self):
        return self.graph.number_of_nodes()
    def edge_count(self):
        return self.graph.number_of_edges()

    def clear(self):
        self.graph.clear()