# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import SlidingWindowMemoryAgnet

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():
    agent = SlidingWindowMemoryAgnet(window_size=4)

    print("=" * 60)
    print("SLIDING WINDOW MEMORY")
    print("=" * 60)

    print("\nWindow size : 4 messages")
    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI : {response}")

        # Showing current memory/token usage
        tracker = agent.get_token_tracker()
        tracker.print_current_report(agent.get_memory())

    # Show complete session statistics
    agent.get_token_tracker().print_total_report()


if __name__ == "__main__":
    main()