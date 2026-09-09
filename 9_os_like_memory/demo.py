# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from .agent import OSMemoryAgent

# =========================================================================================================
#                                           main demo Statements
# =========================================================================================================

def main():

    agent = OSMemoryAgent(ram_size=2)

    print("=" * 60)
    print("OS-LIKE MEMORY MANAGEMENT")
    print("=" * 60)

    print("\nRAM capacity: 2 turns")
    print("Passive storage: Disk simulation")
    print("Paging policy: LRU")
    print("LLM: gpt-4o-mini")
    print("\nType 'exit' to stop.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit","quit"}:
            break

        response = agent.chat( user_input)
        print(f"\nAI: {response}")
        tracker = (agent.get_token_tracker())
        tracker.print_current_report([])
        memory = agent.get_memory()
        stats = (memory.get_statistics())

        print(
            f"\nRAM items: "
            f"{stats['ram_items']}"
        )

        print(
            f"Disk items: "
            f"{stats['disk_items']}"
        )

        print(
            f"Page faults: "
            f"{stats['page_faults']}"
        )

        print(
            f"Pages out: "
            f"{stats['pages_out']}"
        )

        print(
            f"Pages in: "
            f"{stats['pages_in']}"
        )
    agent.get_token_tracker().print_total_report()