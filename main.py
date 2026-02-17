import os
import sys
from src.agent import Agent
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def main():
    print(Fore.CYAN + "Starting AI Agent with Self-Learning Loop...")
    
    try:
        agent = Agent()
    except Exception as e:
        print(Fore.RED + f"Failed to initialize agent: {e}")
        return

    print(Fore.GREEN + "Initialization Complete. Type 'exit' to quit.")
    print(Fore.YELLOW + "Options: Type '/learn' after a response to save it to memory.")
    
    while True:
        try:
            user_input = input(Fore.BLUE + "\nYou: " + Style.RESET_ALL)
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.strip() == "":
                continue

            response = agent.run(user_input)
            
            print(Fore.MAGENTA + f"\nAgent: {response}")
            
            # Simple feedback loop request
            feedback_opt = input(Fore.YELLOW + "\n(Press Enter to continue, or type 'good' to save this interaction): " + Style.RESET_ALL)
            if feedback_opt.lower() in ["good", "save", "yes"]:
                agent.learn(user_input, response, "User validated this response.")
                print(Fore.GREEN + "Interaction saved to long-term memory.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")

if __name__ == "__main__":
    main()
