import os
import sys
from dotenv import load_dotenv

load_dotenv()

# --- Telegram Configuration ---
# These MUST be set in your .env file. There are no insecure hardcoded fallbacks.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        "[Config] WARNING: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID are not set in your .env file. "
        "Telegram features will be disabled.",
        file=sys.stderr
    )

# --- Ollama Configuration ---
_ollama_hosts_raw = os.getenv("OLLAMA_HOSTS", "http://127.0.0.1:11434")
OLLAMA_HOSTS = [h.strip() for h in _ollama_hosts_raw.split(",") if h.strip()]

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "20000"))
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))  # -1 keeps models loaded indefinitely

# --- Agent Configuration ---
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # Seconds between autonomy pulses
MAX_AUTONOMY_STEPS = int(os.getenv("MAX_AUTONOMY_STEPS", "5"))   # Max steps per pulse

# --- Telegram Polling ---
TELEGRAM_POLL_TIMEOUT = int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30"))
TELEGRAM_RETRY_DELAY = int(os.getenv("TELEGRAM_RETRY_DELAY", "1"))

# --- Auto-Learning ---
# Minimum combined character length of user+response to trigger insight extraction.
# Prevents expensive LLM calls on trivial one-liner exchanges.
AUTO_LEARN_MIN_LENGTH = int(os.getenv("AUTO_LEARN_MIN_LENGTH", "200"))
