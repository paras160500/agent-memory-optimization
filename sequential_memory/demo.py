# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import SequentialMemoryAgent

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():
    # Agent creation
    agent = SequentialMemoryAgent()

    print("="*60)
    print("SEQUENTIAL MEMORY")
    print("="*60)

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI : {response}")

        # Show current memory/token usage
        tracker = agent.get_token_tracker()
        tracker.print_current_report(agent.get_memory())

    # Show complete session staistics
    agent.get_token_tracker().print_total_report()

if __name__ == "__main__":
    main()