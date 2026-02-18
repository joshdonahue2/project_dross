import os
import sys
from src.agent_graph import DROSSGraph
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def main():
    print(Fore.CYAN + "Starting AI Agent with Modernized LangGraph Engine...")

    try:
        agent = DROSSGraph()
    except Exception as e:
        print(Fore.RED + f"Failed to initialize agent: {e}")
        return

    print(Fore.GREEN + "Initialization Complete. Type 'exit' to quit.")

    while True:
        try:
            user_input = input(Fore.BLUE + "\nYou: " + Style.RESET_ALL)
            if user_input.lower() in ["exit", "quit"]:
                break

            if user_input.strip() == "":
                continue

            response = agent.run(user_input)

            print(Fore.MAGENTA + f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")

if __name__ == "__main__":
    main()
