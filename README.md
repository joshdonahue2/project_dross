# project_dross

DROSS (Digital Operative & System Sentinel) is a modernized local autonomous AI agent.

## Modernized Architecture

The project has been upgraded from a legacy sequential loop to a robust, state-based orchestration using **LangGraph**.

### Key Components

- **Agent Engine**: Powered by `LangGraph` in `src/agent_graph.py`. It utilizes a state machine for routing, reasoning, and tool execution.
- **Data Management**: Uses a **SQLite** database (`data/dross.db`) for persistent storage of goals, plans, and journal entries, ensuring atomicity and consistency.
- **Validation**: Strict type safety and data validation using **Pydantic** models.
- **Memory**: Hybrid memory system with short-term conversation history and long-term vector storage via **ChromaDB**.
- **Frontend**: Modern **React + Tailwind CSS** interface in `src/static/index.html` (Ether theme), providing real-time visualization of agent state and activity.
- **Backend**: **FastAPI** server with WebSocket support for live streaming of agent operations.

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally or accessible via network.

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your `.env` file (see `.env.example`).

### Running the Agent

#### Start the Web GUI
```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8001
```
Open your browser at `http://localhost:8001`.

#### Run the CLI
```bash
python main.py
```

### Diagnostics

Run the diagnostic script to verify your setup:
```bash
python diagnose.py
```

## Testing

Run the modernized test suite:
```bash
PYTHONPATH=. pytest tests/test_comprehensive.py
```
