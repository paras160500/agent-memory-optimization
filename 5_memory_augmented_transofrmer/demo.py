# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import MemoryAugmentedTransformerAgent

# =========================================================================================================
#                                              main Statements
# =========================================================================================================

def main():
    agent = MemoryAugmentedTransformerAgent(top_k=3)

    print("="*60)
    print("MEMORY)AUGMENTED TRANSFORMER")
    print("="*60)

    print("Tpye 'exit' to stop")
    print("\nAttention Top-k : 3")

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break
        response = agent.chat(user_input)
        print(f"\nAI : {response}")
        tracker = agent.get_token_tracker()
        tracker.print_current_report([])
        print(f"Extrenal memory slots : {agent.get_memory().count()}")
    agent.get_token_tracker().print_total_report()


if __name__ == "__main__":
    main()