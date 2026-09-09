# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import RetrievalMemoryAgent

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():
    agent = RetrievalMemoryAgent(top_k = 3)
    print("="*60)
    print("RETRIEVAL_BASED MEMORY")
    print("="*60)

    print("\nTop-K memories : 3")
    print("Embedding : text-embedding-3-small")
    print("Type exit to stop")

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI : {response}")
        tracker = agent.get_token_tracker()
        tracker.print_current_report([])
        print(
            f"Stored memories"
            f"{agent.get_memory().count()}"
        )
    agent.get_token_tracker().print_total_report()

if __name__ == "__main__":
    main()