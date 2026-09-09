# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import HierarchicalMemoryAgent

# =========================================================================================================
#                                              main Statements
# =========================================================================================================

def main():
    agent = HierarchicalMemoryAgent(4 , 3)
    print("="*60)
    print("HIERARCHICAL MEMORY")
    print("="*60)

    print("\nType exit to stop")
    print("\nLong-term retrieval : top-k=3")
    print("\nWorking memory window : 4 messages")

    while True:
        user_input = input("\nYou :- ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI :- {response}")
        tracker = agent.get_token_tracker()
        tracker.print_current_report()
        memory = agent.get_memory()
        print(f"\nWorking memory messages : {memory.working_count()}")
        print(f"\nLong-term memories : {memory.long_term_count()}")

    agent.get_token_tracker().print_total_report()

if __name__ == "__main__":
    main()