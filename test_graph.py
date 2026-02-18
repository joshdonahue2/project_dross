from src.agent_graph import DROSSGraph
import sys
import os

# Mocking some things if needed, but ModelManager and MemorySystem should work if Ollama is running.
# Given the environment, I assume Ollama is available as per memory.

def test_graph():
    graph = DROSSGraph()
    print("Testing DROSS Graph...")
    user_input = "Hello, who are you?"
    response = graph.run(user_input)
    print(f"User: {user_input}")
    print(f"Assistant: {response}")

if __name__ == "__main__":
    test_graph()
