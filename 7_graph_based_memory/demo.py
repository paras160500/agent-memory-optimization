# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import GraphBaseMemoryAgent

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():
    agent = GraphBaseMemoryAgent()

    print("="*60)
    print("GRAPH-BASED MEMORY")
    print("="*60)

    print("\nType 'exit' to stop")

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI : {response}")
        tracker = agent.get_token_tracker()
        tracker.print_current_report([])
        memory = agent.get_memory()
        print(f"\nGraph nodes : {memory.node_count()}")
        print(f"Graph relationships: {memory.edge_count()}")

    agent.get_token_tracker().print_total_report()

if __name__ == "__main__":
    main()