import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .logger import get_logger

logger = get_logger("db")

DB_PATH = os.path.join("data", "dross.db")

class DROSSDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Goals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_autonomous BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT
                )
            """)
            # Subtasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subtasks (
                    id TEXT PRIMARY KEY,
                    goal_id INTEGER,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals (id) ON DELETE CASCADE
                )
            """)
            # Plans table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (goal_id) REFERENCES goals (id) ON DELETE CASCADE
                )
            """)
            # Plan steps table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER,
                    step_index INTEGER,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES plans (id) ON DELETE CASCADE
                )
            """)
            # Journal table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    entry TEXT NOT NULL
                )
            """)
            # Chat messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            # Activity logs (Live Stream) table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.commit()

    # --- Goal Methods ---

    def set_goal(self, description: str, is_autonomous: bool = False) -> int:
        with self._get_connection() as conn:
            # Mark other active goals as postponed? The original logic pushed to a stack file.
            # For simplicity in this modernization, we'll just have one active goal.
            conn.execute("UPDATE goals SET status = 'postponed' WHERE status = 'active'")

            cursor = conn.execute(
                "INSERT INTO goals (description, status, is_autonomous) VALUES (?, 'active', ?)",
                (description, is_autonomous)
            )
            return cursor.lastrowid

    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY created_at DESC LIMIT 1").fetchone()
            if not row:
                return None
            goal = dict(row)
            # Fetch subtasks
            subtasks = conn.execute("SELECT * FROM subtasks WHERE goal_id = ?", (goal['id'],)).fetchall()
            goal['subtasks'] = [dict(s) for s in subtasks]
            return goal

    def complete_goal(self, goal_id: int, result: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE goals SET status = 'completed', completed_at = ?, result = ? WHERE id = ?",
                (datetime.now().isoformat(), result, goal_id)
            )
            # Try to resume a postponed goal
            postponed = conn.execute("SELECT id FROM goals WHERE status = 'postponed' ORDER BY created_at DESC LIMIT 1").fetchone()
            if postponed:
                conn.execute("UPDATE goals SET status = 'active' WHERE id = ?", (postponed['id'],))

    # --- Plan Methods ---

    def set_plan(self, goal_id: int, steps: List[str]):
        with self._get_connection() as conn:
            # Remove old plan for this goal if exists
            conn.execute("DELETE FROM plans WHERE goal_id = ?", (goal_id,))
            cursor = conn.execute("INSERT INTO plans (goal_id) VALUES (?)", (goal_id,))
            plan_id = cursor.lastrowid

            for i, step in enumerate(steps):
                conn.execute(
                    "INSERT INTO plan_steps (plan_id, step_index, description, status) VALUES (?, ?, ?, 'pending')",
                    (plan_id, i, step)
                )

    def get_plan(self, goal_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM plans WHERE goal_id = ?", (goal_id,)).fetchone()
            if not row:
                return None
            plan = dict(row)
            steps = conn.execute("SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_index ASC", (plan['id'],)).fetchall()
            plan['steps'] = [dict(s) for s in steps]
            return plan

    def update_plan_step(self, plan_id: int, step_index: int, status: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE plan_steps SET status = ? WHERE plan_id = ? AND step_index = ?",
                (status, plan_id, step_index)
            )

    # --- Journal Methods ---

    def add_journal_entry(self, entry: str):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO journal (entry) VALUES (?)", (entry,))

    def get_journal_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM journal ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    # --- Message & Activity Methods ---

    def add_message(self, role: str, content: str):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))

    def get_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM messages ORDER BY id ASC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def add_activity_log(self, log_type: str, content: str):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO activity_logs (type, content) VALUES (?, ?)", (log_type, content))

    def get_activity_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM activity_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def clear_ui_state(self):
        """Clears chat and logs (e.g. on system reset)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM activity_logs")
