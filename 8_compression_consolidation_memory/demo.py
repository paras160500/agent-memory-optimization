# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import CompressionMemoryAgent

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():

    agent = CompressionMemoryAgent()
    print("=" * 60)
    print("COMPRESSION & CONSOLIDATION MEMORY")
    print("=" * 60)
    print("\nMemory type: Dense factual compression")
    print("LLM: gpt-4o-mini")
    print("\nType 'exit' to stop.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit","quit"}:
            break
        response = agent.chat(user_input)
        print(f"\nAI: {response}")

        tracker = (agent.get_token_tracker())
        tracker.print_current_report([])
        print(
            f"\nCompressed facts stored: "
            f"{agent.get_memory().count()}"
        )

    agent.get_token_tracker().print_total_report()

if __name__ == "__main__":
    main()