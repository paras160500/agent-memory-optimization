# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import SummarizationMemoryAgent

# =========================================================================================================
#                                              Class Statements
# =========================================================================================================

def main():
    agent = SummarizationMemoryAgent(max_recent_messages=3)

    print("="*60)
    print("SUMMARIZATION MEMORY")
    print("="*60)

    print("\nRecent messages kept : 4")
    print("Type exit to quit")

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() in {"exit" , "quit"}:
            break 
        response = agent.chat(user_input)
        print(f"\nAI : " , {response})

        tracker = agent.get_token_tracker()
        tracker.print_current_report(agent.get_memory().get_recent_messages())

        print("\nCurrent Summary")
        print(agent.get_memory().get_summary())

    agent.get_token_tracker().print_total_report()


if __name__ == "__main__":
    main()