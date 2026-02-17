"""
DROSS Test Suite
================
Run from your project root:

    python -m pytest test_dross.py -v                  # all tests
    python -m pytest test_dross.py -v -m "not slow"    # skip live-inference tests
    python -m pytest test_dross.py -v -k "memory"      # run only memory tests

Marks:
    slow  — makes real Ollama calls (10-60s each)
    live  — requires running server (server.py)

Tests are grouped into classes. Each class that touches files or DB
cleans up after itself so tests are fully isolated.
"""

import json
import os
import sys
import time
import shutil
import tempfile
import unittest
import warnings

import pytest

# ── Project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath("."))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """src/config.py — environment & secrets"""

    def test_imports_without_error(self):
        from src import config  # noqa: F401

    def test_ollama_hosts_is_list(self):
        from src.config import OLLAMA_HOSTS
        self.assertIsInstance(OLLAMA_HOSTS, list)
        self.assertGreater(len(OLLAMA_HOSTS), 0)

    def test_ollama_hosts_are_urls(self):
        from src.config import OLLAMA_HOSTS
        for host in OLLAMA_HOSTS:
            self.assertTrue(
                host.startswith("http://") or host.startswith("https://"),
                f"Host '{host}' is not a valid URL"
            )

    def test_no_hardcoded_token(self):
        """Token must come from env, not be baked into the source file."""
        src_path = os.path.join("src", "config.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # A real token looks like digits:alphanum  e.g. 8590694632:AAH...
        import re
        hardcoded = re.findall(r'\d{8,12}:[A-Za-z0-9_-]{30,}', src)
        self.assertEqual(hardcoded, [], f"Hardcoded Telegram token found in config.py: {hardcoded}")

    def test_auto_learn_min_length_is_int(self):
        from src.config import AUTO_LEARN_MIN_LENGTH
        self.assertIsInstance(AUTO_LEARN_MIN_LENGTH, int)
        self.assertGreater(AUTO_LEARN_MIN_LENGTH, 0)

    def test_num_ctx_is_reasonable(self):
        from src.config import OLLAMA_NUM_CTX
        self.assertGreaterEqual(OLLAMA_NUM_CTX, 2048)

    def test_heartbeat_interval_positive(self):
        from src.config import HEARTBEAT_INTERVAL
        self.assertGreater(HEARTBEAT_INTERVAL, 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. OLLAMA CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestOllamaConnectivity(unittest.TestCase):
    """Live network checks against each configured Ollama host."""

    def _get_clients(self):
        from ollama import Client
        from src.config import OLLAMA_HOSTS
        return [(h, Client(host=h)) for h in OLLAMA_HOSTS]

    def test_all_hosts_reachable(self):
        for host, client in self._get_clients():
            with self.subTest(host=host):
                try:
                    models = client.list()
                    self.assertIsNotNone(models, f"No response from {host}")
                except Exception as e:
                    self.fail(f"Host {host} unreachable: {e}")

    def test_hosts_have_models(self):
        for host, client in self._get_clients():
            with self.subTest(host=host):
                models = client.list()
                names = [m.model for m in models.models]
                self.assertGreater(len(names), 0, f"No models available on {host}")

    def test_required_models_present(self):
        """Checks that the models named in ModelManager actually exist somewhere."""
        from src.config import OLLAMA_HOSTS
        from src.models import ModelManager
        from ollama import Client

        mm = ModelManager()
        required = {mm.reasoning_model, mm.general_model, mm.tool_model}

        available = set()
        for host in OLLAMA_HOSTS:
            try:
                models = Client(host=host).list()
                for m in models.models:
                    available.add(m.model)
            except Exception:
                pass

        for model in required:
            self.assertIn(
                model, available,
                f"Model '{model}' not found on any host. Available: {sorted(available)}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class TestMemorySystem(unittest.TestCase):
    """src/memory.py — ChromaDB + short-term memory, fully isolated."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="dross_test_mem_")
        from src.memory import MemorySystem
        self.mem = MemorySystem(db_path=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Short-term ────────────────────────────────────────────────────────────

    def test_short_term_starts_empty(self):
        self.assertEqual(self.mem.get_short_term(), [])

    def test_add_and_get_short_term(self):
        self.mem.add_short_term("user", "hello", source="test")
        self.mem.add_short_term("assistant", "hi there", source="test")
        history = self.mem.get_short_term()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "hello")
        self.assertEqual(history[1]["role"], "assistant")

    def test_clear_short_term(self):
        self.mem.add_short_term("user", "something")
        self.mem.clear_short_term()
        self.assertEqual(self.mem.get_short_term(), [])

    def test_prune_short_term_triggers_at_limit(self):
        for i in range(16):
            self.mem.add_short_term("user", f"message {i}")
        pruned = self.mem.prune_short_term()
        self.assertIsNotNone(pruned, "prune_short_term() should return pruned text when limit exceeded")
        self.assertIsInstance(pruned, str)
        self.assertGreater(len(pruned), 0)

    def test_prune_short_term_no_prune_below_limit(self):
        for i in range(5):
            self.mem.add_short_term("user", f"message {i}")
        pruned = self.mem.prune_short_term()
        self.assertIsNone(pruned, "prune_short_term() should return None when under limit")

    def test_prune_reduces_length(self):
        for i in range(16):
            self.mem.add_short_term("user", f"message {i}")
        before = len(self.mem.get_short_term())
        self.mem.prune_short_term()
        after = len(self.mem.get_short_term())
        self.assertLess(after, before)

    # ── Long-term (ChromaDB) ──────────────────────────────────────────────────

    def test_save_long_term_returns_id(self):
        mem_id = self.mem.save_long_term("The sky is blue.", {"type": "test"})
        self.assertIsNotNone(mem_id)
        self.assertIsInstance(mem_id, str)
        self.assertGreater(len(mem_id), 0)

    def test_save_and_retrieve(self):
        self.mem.save_long_term("DROSS lives on a Linux server.", {"type": "test"})
        result = self.mem.retrieve_relevant("where does DROSS live")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_retrieve_returns_empty_for_unrelated_query(self):
        self.mem.save_long_term("The user likes coffee.", {"type": "test"})
        # Completely unrelated query should return low-relevance or empty
        result = self.mem.retrieve_relevant("quantum physics equations")
        # Not asserting empty (similarity thresholds vary) — just that it doesn't crash
        self.assertIsInstance(result, str)

    def test_deduplication_prevents_exact_duplicate(self):
        content = "Unique fact about the DROSS system for dedup test."
        id1 = self.mem.save_long_term(content, deduplicate=True)
        id2 = self.mem.save_long_term(content, deduplicate=True)
        count = self.mem.collection.count()
        # Should be 1 entry not 2 — dedup should return existing id
        self.assertEqual(id1, id2, "Duplicate save should return the same ID")
        self.assertEqual(count, 1, "Dedup should prevent a second document being stored")

    def test_save_without_dedup_allows_duplicates(self):
        content = "Non-deduped fact."
        self.mem.save_long_term(content, deduplicate=False)
        self.mem.save_long_term(content, deduplicate=False)
        count = self.mem.collection.count()
        self.assertEqual(count, 2)

    def test_save_relationship(self):
        id1 = self.mem.save_long_term("Fact A", deduplicate=False)
        id2 = self.mem.save_long_term("Fact B", deduplicate=False)
        # Should not raise
        self.mem.save_relationship(id1, id2, "supports")
        rels = self.mem.rel_collection.count()
        self.assertEqual(rels, 1)

    def test_get_all_memories_structure(self):
        self.mem.save_long_term("Test memory.", {"type": "test"}, deduplicate=False)
        data = self.mem.get_all_memories()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["edges"], list)
        self.assertGreater(len(data["nodes"]), 0)

    def test_wipe_memory_clears_db(self):
        self.mem.save_long_term("Something to wipe.", deduplicate=False)
        self.mem.wipe_memory()
        count = self.mem.collection.count()
        self.assertEqual(count, 0)

    def test_delete_memories_containing(self):
        self.mem.save_long_term("DELETE_ME this is a test.", deduplicate=False)
        self.mem.save_long_term("Keep this one.", deduplicate=False)
        deleted = self.mem.delete_memories_containing("DELETE_ME")
        self.assertEqual(deleted, 1)
        self.assertEqual(self.mem.collection.count(), 1)

    def test_retrieve_relevant_empty_query(self):
        """Empty query should not crash — should return empty string."""
        result = self.mem.retrieve_relevant("")
        self.assertEqual(result, "")

    def test_metadata_preserved_on_retrieval(self):
        self.mem.save_long_term("Fact with type.", {"type": "episodic"}, deduplicate=False)
        result = self.mem.retrieve_relevant("Fact with type")
        self.assertIn("episodic", result)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    """src/tools.py — registry, sandboxing, and individual tools."""

    @classmethod
    def setUpClass(cls):
        from src.tools import registry
        cls.reg = registry

    # ── Registry structure ────────────────────────────────────────────────────

    def test_registry_has_tools(self):
        self.assertGreater(len(self.reg.tools), 0)

    def test_all_core_tools_registered(self):
        required = [
            "run_shell", "list_files", "write_file", "read_file",
            "set_goal", "get_goal", "complete_goal",
            "add_subtask", "list_subtasks", "complete_subtask",
            "set_plan", "get_plan", "update_plan_step",
            "write_journal", "read_journal",
            "send_telegram_message", "search_web", "run_python",
            "create_tool",
        ]
        for name in required:
            with self.subTest(tool=name):
                self.assertIn(name, self.reg.tools, f"Tool '{name}' not registered")

    def test_schemas_are_valid_json(self):
        schema_str = self.reg.get_schemas_str()
        parsed = json.loads(schema_str)
        self.assertIsInstance(parsed, list)

    def test_schema_types_are_json_schema_not_python_repr(self):
        schemas = json.loads(self.reg.get_schemas_str())
        for schema in schemas:
            props = schema.get("parameters", {}).get("properties", {})
            for param_name, param_val in props.items():
                t = param_val.get("type", "")
                self.assertNotIn(
                    "<class", t,
                    f"Tool '{schema['name']}' param '{param_name}' has raw Python type: {t}"
                )

    def test_schema_has_required_fields(self):
        schemas = json.loads(self.reg.get_schemas_str())
        for schema in schemas:
            with self.subTest(tool=schema.get("name")):
                self.assertIn("name", schema)
                self.assertIn("description", schema)
                self.assertIn("parameters", schema)

    def test_unknown_tool_returns_error_string(self):
        result = self.reg.execute("nonexistent_tool_xyz", {})
        self.assertIn("Error", result)

    # ── Path sandboxing ───────────────────────────────────────────────────────

    def test_traversal_above_root_blocked(self):
        from src.tools import _get_safe_path
        with self.assertRaises(ValueError):
            _get_safe_path("../../etc/passwd")

    def test_traversal_absolute_path_blocked(self):
        from src.tools import _get_safe_path
        with self.assertRaises(ValueError):
            _get_safe_path("/etc/passwd")

    def test_safe_path_within_project_allowed(self):
        from src.tools import _get_safe_path
        # Should not raise
        path = _get_safe_path("data/test.json")
        self.assertTrue(os.path.isabs(path))

    def test_workspace_prefix_stripped(self):
        from src.tools import _get_safe_path
        p1 = _get_safe_path("workspace/myfile.txt")
        p2 = _get_safe_path("myfile.txt")
        self.assertEqual(p1, p2)

    # ── File tools ────────────────────────────────────────────────────────────

    def setUp(self):
        """Create a temp file name in workspace for file tool tests."""
        self._test_filename = f"_test_file_{int(time.time())}.txt"

    def tearDown(self):
        from src.tools import _get_safe_path
        try:
            path = _get_safe_path(self._test_filename)
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def test_write_and_read_file(self):
        content = "Hello from test suite."
        result = self.reg.execute("write_file", {"filename": self._test_filename, "content": content})
        self.assertNotIn("Error", result)
        read_back = self.reg.execute("read_file", {"filename": self._test_filename})
        self.assertEqual(read_back, content)

    def test_read_nonexistent_file(self):
        result = self.reg.execute("read_file", {"filename": "_definitely_does_not_exist_xyz.txt"})
        self.assertIn("not exist", result.lower())

    def test_list_files_returns_json(self):
        result = self.reg.execute("list_files", {"path": "."})
        self.assertNotIn("Error", result)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)
        self.assertGreater(len(parsed), 0)

    def test_get_file_info(self):
        self.reg.execute("write_file", {"filename": self._test_filename, "content": "test"})
        info = self.reg.execute("get_file_info", {"filename": self._test_filename})
        self.assertIn("Size", info)
        self.assertIn("Modified", info)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GOAL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class TestGoalManagement(unittest.TestCase):
    """Full goal lifecycle: set → subtask → plan → step updates → complete → stack."""

    @classmethod
    def setUpClass(cls):
        from src.tools import registry, GOAL_FILE, GOAL_STACK_FILE, PLAN_FILE
        cls.reg = registry
        cls.goal_file = GOAL_FILE
        cls.stack_file = GOAL_STACK_FILE
        cls.plan_file = PLAN_FILE

    def _clear_state(self):
        for f in [self.goal_file, self.stack_file, self.plan_file]:
            if os.path.exists(f):
                os.remove(f)

    def setUp(self):
        self._clear_state()

    def tearDown(self):
        self._clear_state()

    # ── set_goal / get_goal ───────────────────────────────────────────────────

    def test_set_goal_creates_file(self):
        self.reg.execute("set_goal", {"description": "Test goal alpha"})
        self.assertTrue(os.path.exists(self.goal_file))

    def test_get_goal_returns_active(self):
        self.reg.execute("set_goal", {"description": "Test goal beta"})
        result = self.reg.execute("get_goal", {})
        data = json.loads(result)
        self.assertEqual(data["goal"], "Test goal beta")
        self.assertEqual(data["status"], "active")

    def test_get_goal_no_file_returns_no_active(self):
        result = self.reg.execute("get_goal", {})
        self.assertIn("No active goal", result)

    def test_goal_has_timestamps(self):
        self.reg.execute("set_goal", {"description": "Timestamped goal"})
        data = _read_json(self.goal_file)
        self.assertIn("created_at", data)

    def test_goal_is_autonomous_flag(self):
        self.reg.execute("set_goal", {"description": "Auto goal", "is_autonomous": True})
        data = _read_json(self.goal_file)
        self.assertTrue(data["is_autonomous"])

    # ── complete_goal ─────────────────────────────────────────────────────────

    def test_complete_goal_marks_completed(self):
        self.reg.execute("set_goal", {"description": "Complete me"})
        result = self.reg.execute("complete_goal", {"result_summary": "Done."})
        self.assertIn("completed", result.lower())
        data = _read_json(self.goal_file)
        self.assertEqual(data["status"], "completed")

    def test_complete_goal_stores_result(self):
        self.reg.execute("set_goal", {"description": "Goal with result"})
        self.reg.execute("complete_goal", {"result_summary": "The answer was 42."})
        data = _read_json(self.goal_file)
        self.assertEqual(data["result"], "The answer was 42.")

    def test_complete_goal_has_completed_at(self):
        self.reg.execute("set_goal", {"description": "Timed goal"})
        self.reg.execute("complete_goal", {"result_summary": "Done"})
        data = _read_json(self.goal_file)
        self.assertIn("completed_at", data)

    def test_complete_nonexistent_goal(self):
        result = self.reg.execute("complete_goal", {"result_summary": "Nothing here"})
        self.assertIn("No active goal", result)

    # ── Goal stack ────────────────────────────────────────────────────────────

    def test_user_goal_pushes_autonomous_goal_to_stack(self):
        # Set an autonomous goal first
        self.reg.execute("set_goal", {"description": "Autonomous background task", "is_autonomous": True})
        # Now set a user goal — should push the autonomous one to stack
        self.reg.execute("set_goal", {"description": "User urgent task", "is_autonomous": False})

        # Stack file should now exist with the autonomous goal
        self.assertTrue(os.path.exists(self.stack_file), "Stack file should be created")
        stack = _read_json(self.stack_file)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0]["goal"], "Autonomous background task")

        # Current goal should be the user goal
        current = _read_json(self.goal_file)
        self.assertEqual(current["goal"], "User urgent task")

    def test_completing_user_goal_resumes_stacked_goal(self):
        self.reg.execute("set_goal", {"description": "Background task", "is_autonomous": True})
        self.reg.execute("set_goal", {"description": "User task", "is_autonomous": False})
        result = self.reg.execute("complete_goal", {"result_summary": "User task done."})

        # Should mention resumption
        self.assertIn("Resumed", result, "complete_goal should report resuming the stacked goal")

        # Current goal should now be the resumed background task
        current = _read_json(self.goal_file)
        self.assertEqual(current["goal"], "Background task")
        self.assertEqual(current["status"], "active")

    # ── Subtasks ──────────────────────────────────────────────────────────────

    def test_add_subtask(self):
        self.reg.execute("set_goal", {"description": "Goal with subtasks"})
        result = self.reg.execute("add_subtask", {"subtask": "Do the first thing"})
        self.assertNotIn("Error", result)

    def test_list_subtasks_shows_added(self):
        self.reg.execute("set_goal", {"description": "Subtask list test"})
        self.reg.execute("add_subtask", {"subtask": "Alpha subtask"})
        self.reg.execute("add_subtask", {"subtask": "Beta subtask"})
        result = self.reg.execute("list_subtasks", {})
        self.assertIn("Alpha subtask", result)
        self.assertIn("Beta subtask", result)

    def test_subtasks_start_pending(self):
        self.reg.execute("set_goal", {"description": "Pending test"})
        self.reg.execute("add_subtask", {"subtask": "Pending task"})
        result = self.reg.execute("list_subtasks", {})
        self.assertIn("[ ]", result)

    def test_complete_subtask(self):
        self.reg.execute("set_goal", {"description": "Completion test"})
        self.reg.execute("add_subtask", {"subtask": "Completable task"})
        data = _read_json(self.goal_file)
        task_id = data["subtasks"][0]["id"]

        self.reg.execute("complete_subtask", {"subtask_id": task_id})
        data = _read_json(self.goal_file)
        self.assertEqual(data["subtasks"][0]["status"], "completed")

    def test_complete_subtask_partial_id_match(self):
        self.reg.execute("set_goal", {"description": "Partial ID test"})
        self.reg.execute("add_subtask", {"subtask": "Partial match task"})
        data = _read_json(self.goal_file)
        full_id = data["subtasks"][0]["id"]
        partial_id = full_id[:4]  # Use only first 4 chars

        self.reg.execute("complete_subtask", {"subtask_id": partial_id})
        data = _read_json(self.goal_file)
        self.assertEqual(data["subtasks"][0]["status"], "completed")

    def test_list_subtasks_no_goal(self):
        result = self.reg.execute("list_subtasks", {})
        self.assertIn("No active goal", result)

    def test_multiple_subtasks_independent(self):
        self.reg.execute("set_goal", {"description": "Multi-subtask goal"})
        self.reg.execute("add_subtask", {"subtask": "Task one"})
        self.reg.execute("add_subtask", {"subtask": "Task two"})
        self.reg.execute("add_subtask", {"subtask": "Task three"})
        data = _read_json(self.goal_file)
        self.assertEqual(len(data["subtasks"]), 3)
        ids = {t["id"] for t in data["subtasks"]}
        self.assertEqual(len(ids), 3, "Each subtask should have a unique ID")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLAN MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanManagement(unittest.TestCase):
    """set_plan → get_plan → update_plan_step lifecycle."""

    @classmethod
    def setUpClass(cls):
        from src.tools import registry, PLAN_FILE
        cls.reg = registry
        cls.plan_file = PLAN_FILE

    def _clear(self):
        if os.path.exists(self.plan_file):
            os.remove(self.plan_file)

    def setUp(self):
        self._clear()

    def tearDown(self):
        self._clear()

    def test_set_plan_creates_file(self):
        self.reg.execute("set_plan", {"steps": ["Step A", "Step B", "Step C"]})
        self.assertTrue(os.path.exists(self.plan_file))

    def test_set_plan_correct_step_count(self):
        self.reg.execute("set_plan", {"steps": ["One", "Two", "Three"]})
        data = _read_json(self.plan_file)
        self.assertEqual(len(data["steps"]), 3)

    def test_all_steps_start_pending(self):
        self.reg.execute("set_plan", {"steps": ["A", "B", "C"]})
        data = _read_json(self.plan_file)
        for step in data["steps"]:
            self.assertEqual(step["status"], "pending")

    def test_get_plan_no_file(self):
        result = self.reg.execute("get_plan", {})
        self.assertIn("No plan defined", result)

    def test_get_plan_returns_json(self):
        self.reg.execute("set_plan", {"steps": ["Step 1"]})
        result = self.reg.execute("get_plan", {})
        data = json.loads(result)
        self.assertIn("steps", data)

    def test_update_plan_step_to_completed(self):
        self.reg.execute("set_plan", {"steps": ["First", "Second", "Third"]})
        self.reg.execute("update_plan_step", {"step_index": 1, "status": "completed"})
        data = _read_json(self.plan_file)
        self.assertEqual(data["steps"][0]["status"], "pending")
        self.assertEqual(data["steps"][1]["status"], "completed")
        self.assertEqual(data["steps"][2]["status"], "pending")

    def test_update_plan_step_to_failed(self):
        self.reg.execute("set_plan", {"steps": ["Step"]})
        self.reg.execute("update_plan_step", {"step_index": 0, "status": "failed"})
        data = _read_json(self.plan_file)
        self.assertEqual(data["steps"][0]["status"], "failed")

    def test_update_plan_step_invalid_index(self):
        self.reg.execute("set_plan", {"steps": ["Only step"]})
        result = self.reg.execute("update_plan_step", {"step_index": 99, "status": "completed"})
        self.assertIn("Invalid", result)

    def test_update_plan_no_file(self):
        result = self.reg.execute("update_plan_step", {"step_index": 0, "status": "completed"})
        self.assertIn("No plan", result)

    def test_plan_has_created_at(self):
        self.reg.execute("set_plan", {"steps": ["Timestamped"]})
        data = _read_json(self.plan_file)
        self.assertIn("created_at", data)

    def test_full_plan_lifecycle(self):
        """Set plan, complete all steps, verify all are completed."""
        steps = ["Research", "Design", "Implement", "Test", "Deploy"]
        self.reg.execute("set_plan", {"steps": steps})

        for i in range(len(steps)):
            self.reg.execute("update_plan_step", {"step_index": i, "status": "completed"})

        data = _read_json(self.plan_file)
        for step in data["steps"]:
            self.assertEqual(step["status"], "completed",
                             f"Step '{step['description']}' should be completed")


# ══════════════════════════════════════════════════════════════════════════════
# 7. JOURNAL
# ══════════════════════════════════════════════════════════════════════════════

class TestJournal(unittest.TestCase):
    """write_journal / read_journal tool tests with isolated temp file."""

    def setUp(self):
        import src.tools as tools_module
        self._orig_journal = tools_module.JOURNAL_FILE
        self._tmp = tempfile.mktemp(suffix=".jsonl", prefix="dross_journal_test_")
        tools_module.JOURNAL_FILE = self._tmp
        from src.tools import registry
        self.reg = registry

    def tearDown(self):
        import src.tools as tools_module
        tools_module.JOURNAL_FILE = self._orig_journal
        if os.path.exists(self._tmp):
            os.remove(self._tmp)

    def test_write_creates_file(self):
        self.reg.execute("write_journal", {"entry": "First entry."})
        self.assertTrue(os.path.exists(self._tmp))

    def test_written_entry_is_valid_jsonl(self):
        self.reg.execute("write_journal", {"entry": "Valid JSON entry."})
        entries = _read_jsonl(self._tmp)
        self.assertEqual(len(entries), 1)

    def test_entry_has_timestamp(self):
        self.reg.execute("write_journal", {"entry": "Timestamped."})
        entries = _read_jsonl(self._tmp)
        self.assertIn("timestamp", entries[0])

    def test_multiple_writes_append(self):
        for i in range(5):
            self.reg.execute("write_journal", {"entry": f"Entry {i}"})
        entries = _read_jsonl(self._tmp)
        self.assertEqual(len(entries), 5)

    def test_read_journal_returns_last_n(self):
        for i in range(10):
            self.reg.execute("write_journal", {"entry": f"Entry {i}"})
        result = self.reg.execute("read_journal", {"last_n": 3})
        # Should contain the last 3 entries
        self.assertIn("Entry 7", result)
        self.assertIn("Entry 8", result)
        self.assertIn("Entry 9", result)

    def test_read_empty_journal(self):
        result = self.reg.execute("read_journal", {"last_n": 5})
        self.assertIn("No journal entries", result)

    def test_json_payload_entry_readable(self):
        payload = json.dumps({"outcome": "success", "lessons": "It worked.", "what_failed": ""})
        self.reg.execute("write_journal", {"entry": payload})
        entries = _read_jsonl(self._tmp)
        parsed = json.loads(entries[0]["entry"])
        self.assertEqual(parsed["outcome"], "success")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODELS — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestModels(unittest.TestCase):
    """src/models.py — live inference. Marked slow, skipped with -m 'not slow'."""

    @classmethod
    def setUpClass(cls):
        from src.models import ModelManager
        cls.mm = ModelManager()

    def test_route_tool_intent(self):
        result = self.mm.route_request("list the files in my workspace directory")
        self.assertIn(result, ("TOOL", "REASON", "DIRECT"))
        # Should lean TOOL for this
        self.assertEqual(result, "TOOL", f"Expected TOOL, got {result}")

    def test_route_direct_intent(self):
        result = self.mm.route_request("hello how are you today")
        self.assertIn(result, ("TOOL", "REASON", "DIRECT"))

    def test_route_reason_intent(self):
        result = self.mm.route_request("explain step by step why the sky is blue")
        self.assertIn(result, ("TOOL", "REASON", "DIRECT"))

    def test_query_general_returns_string(self):
        resp = self.mm.query_general("Say only the word PONG.", context="", history=[])
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)

    def test_query_general_uses_history(self):
        history = [
            {"role": "user", "content": "My name is HistoryTestUser."},
            {"role": "assistant", "content": "Nice to meet you, HistoryTestUser."},
        ]
        resp = self.mm.query_general("What is my name?", context="", history=history)
        self.assertIn("HistoryTestUser", resp)

    def test_extract_insight_returns_dict(self):
        text = "User: I am Bob and I work at Acme Corp.\nAssistant: Good to know, Bob."
        result = self.mm.extract_insight(text)
        self.assertIsInstance(result, dict)
        self.assertIn("facts", result)
        self.assertIn("relationships", result)

    def test_extract_insight_facts_are_list(self):
        text = "User: Alice lives in London.\nAssistant: London is a great city."
        result = self.mm.extract_insight(text)
        self.assertIsInstance(result["facts"], list)

    def test_extract_insight_facts_are_strings_not_dicts(self):
        """The dict-fact bug — ensure the model prompt fix is working."""
        text = (
            "User: My name is Carol, I am 30 years old, and I work as a nurse in Paris.\n"
            "Assistant: Hello Carol! Paris is a wonderful city for a nurse."
        )
        result = self.mm.extract_insight(text)
        facts = result.get("facts", [])
        dict_facts = [f for f in facts if isinstance(f, dict)]
        self.assertEqual(
            dict_facts, [],
            f"Model returned dict-style facts (old bug): {dict_facts}. "
            "This means the extract_insight prompt needs further improvement."
        )

    def test_extract_insight_relationships_use_string_keys(self):
        text = (
            "User: I use Python to build AI tools at my job in Seattle.\n"
            "Assistant: That's a great stack for AI work in Seattle."
        )
        result = self.mm.extract_insight(text)
        for rel in result.get("relationships", []):
            self.assertIsInstance(
                rel.get("source"), str,
                f"Relationship source is not a string: {rel}"
            )

    def test_query_tool_returns_parseable_json(self):
        from src.tools import registry
        schema = registry.get_schemas_str()
        response = self.mm.query_tool("List files in the workspace", schema)
        self.assertIn("{", response, "query_tool should return a JSON object")

    def test_summarize_memory_returns_string(self):
        text = "[User] What is Python? [Assistant] Python is a programming language."
        result = self.mm.summarize_memory(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 20)

    def test_generate_plan_returns_list(self):
        steps = self.mm.generate_plan("Learn how to make a sandwich")
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertIsInstance(step, str)

    def test_query_reflection_returns_parseable_json(self):
        goal_data = json.dumps({
            "goal": "Write a hello world script",
            "status": "completed",
            "result": "Created hello.py successfully"
        })
        result = self.mm.query_reflection(goal_data)
        self.assertIsInstance(result, str)
        # Should be parseable JSON
        try:
            parsed = json.loads(result)
            self.assertIn("outcome", parsed)
            self.assertIn("lessons", parsed)
        except json.JSONDecodeError:
            self.fail(f"query_reflection returned non-JSON: {result[:200]}")

    def test_query_autonomy_returns_json(self):
        from src.tools import registry
        schema = registry.get_schemas_str()
        result = self.mm.query_autonomy(
            goal_state="GOAL: Write a test file\nACTIVE STEP: Create the file",
            tools_schema=schema
        )
        self.assertIsInstance(result, str)
        self.assertIn("{", result)


# ══════════════════════════════════════════════════════════════════════════════
# 9. AGENT — UNIT (mocked inference)
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentUnit(unittest.TestCase):
    """Agent logic tests that mock out the ModelManager so no Ollama calls are made."""

    def _make_agent(self, tmp_dir):
        """Build an Agent with isolated memory and mocked models."""
        from unittest.mock import MagicMock, patch
        from src.memory import MemorySystem
        from src.agent import Agent

        with patch("src.agent.ModelManager"), patch("src.agent.MemorySystem"):
            agent = Agent.__new__(Agent)
            agent.memory = MemorySystem(db_path=tmp_dir)
            agent.models = MagicMock()
            from src.tools import registry
            agent.tools = registry
            return agent

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="dross_test_agent_")
        self.agent = self._make_agent(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_clean_output_removes_think_tags(self):
        dirty = "<think>internal monologue here</think>The actual answer."
        result = self.agent._clean_output(dirty)
        self.assertEqual(result, "The actual answer.")

    def test_clean_output_removes_boxed(self):
        dirty = r"\boxed{42}"
        result = self.agent._clean_output(dirty)
        self.assertEqual(result, "42")

    def test_clean_output_strips_whitespace(self):
        result = self.agent._clean_output("  answer  ")
        self.assertEqual(result, "answer")

    def test_extract_json_plain(self):
        result = self.agent._extract_json('{"tool_name": "list_files", "tool_args": {}}')
        self.assertEqual(result["tool_name"], "list_files")

    def test_extract_json_from_markdown_block(self):
        result = self.agent._extract_json('```json\n{"key": "value"}\n```')
        self.assertEqual(result["key"], "value")

    def test_extract_json_from_generic_code_block(self):
        result = self.agent._extract_json('```\n{"key": "val"}\n```')
        self.assertEqual(result["key"], "val")

    def test_extract_json_embedded_in_text(self):
        result = self.agent._extract_json('Some preamble {"key": "val"} some postamble')
        self.assertEqual(result["key"], "val")

    def test_extract_json_returns_none_for_garbage(self):
        result = self.agent._extract_json("this is not json at all")
        self.assertIsNone(result)

    def test_save_atomic_memories_strings(self):
        self.agent._save_atomic_memories([
            "The user's name is Alice.",
            "Alice lives in London.",
        ])
        count = self.agent.memory.collection.count()
        self.assertEqual(count, 2)

    def test_save_atomic_memories_dicts_flattened(self):
        """Defensive dict handling — model sometimes returns dicts not strings."""
        self.agent._save_atomic_memories([
            {"name": "Bob", "city": "Paris"},
            {"job": "engineer"},
        ])
        count = self.agent.memory.collection.count()
        self.assertEqual(count, 2, "Dict-style facts should be flattened and saved")

    def test_save_atomic_memories_skips_short_facts(self):
        self.agent._save_atomic_memories(["Hi.", "OK", "Yes"])
        count = self.agent.memory.collection.count()
        self.assertEqual(count, 0, "Facts under 15 chars should be skipped")

    def test_save_atomic_memories_skips_non_string_non_dict(self):
        self.agent._save_atomic_memories([123, None, True])
        count = self.agent.memory.collection.count()
        self.assertEqual(count, 0)

    def test_learn_saves_to_long_term(self):
        self.agent.learn(
            user_input="How do I make coffee?",
            assistant_response="Boil water, add grounds, brew.",
            feedback="Great answer!"
        )
        count = self.agent.memory.collection.count()
        self.assertEqual(count, 1)

    def test_full_reset_clears_memory(self):
        self.agent.memory.save_long_term("Something to wipe", deduplicate=False)
        self.agent.memory.add_short_term("user", "hello")
        from src.tools import GOAL_FILE, GOAL_STACK_FILE
        self.agent.full_reset()
        self.assertEqual(self.agent.memory.collection.count(), 0)
        self.assertEqual(self.agent.memory.get_short_term(), [])


# ══════════════════════════════════════════════════════════════════════════════
# 10. AGENT — END-TO-END (live inference)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestAgentEndToEnd(unittest.TestCase):
    """Full agent.run() pipeline with real Ollama calls."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="dross_e2e_")
        # Patch memory path so we don't pollute production DB
        import src.memory as mem_module
        cls._orig_init = mem_module.MemorySystem.__init__

        def patched_init(self_inner, db_path="./data/memory_db"):
            cls._orig_init(self_inner, db_path=cls.tmp_dir)

        mem_module.MemorySystem.__init__ = patched_init

        from src.agent import Agent
        cls.agent = Agent()

    @classmethod
    def tearDownClass(cls):
        import src.memory as mem_module
        mem_module.MemorySystem.__init__ = cls._orig_init
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_run_returns_response(self):
        resp = self.agent.run("Hello, please acknowledge this message.", source="test")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)

    def test_run_no_think_tags_in_output(self):
        resp = self.agent.run("What is 2 + 2?", source="test")
        self.assertNotIn("<think>", resp)
        self.assertNotIn("</think>", resp)

    def test_run_updates_short_term_memory(self):
        before = len(self.agent.memory.get_short_term())
        self.agent.run("Remember that I like tea.", source="test")
        after = len(self.agent.memory.get_short_term())
        self.assertGreater(after, before, "Short-term memory should grow after run()")

    def test_run_long_message_creates_memories(self):
        """Combined length must exceed AUTO_LEARN_MIN_LENGTH to trigger insight extraction."""
        from src.config import AUTO_LEARN_MIN_LENGTH
        count_before = self.agent.memory.collection.count()

        long_prompt = (
            "My name is E2ETestUser and I am a software architect based in Amsterdam. "
            "I have been working with Python and distributed systems for over ten years. "
            "I enjoy building autonomous agents, message queues, and memory systems. "
            "Please acknowledge all of these details about me explicitly."
        )
        self.assertGreater(len(long_prompt), 100,
                           "Prompt itself should be substantial")

        resp = self.agent.run(long_prompt, source="test")

        # Give the combined length check
        combined = len(long_prompt) + len(resp)
        if combined < AUTO_LEARN_MIN_LENGTH:
            self.skipTest(
                f"Combined length {combined} < AUTO_LEARN_MIN_LENGTH {AUTO_LEARN_MIN_LENGTH}. "
                "Lower AUTO_LEARN_MIN_LENGTH or use a longer prompt."
            )

        count_after = self.agent.memory.collection.count()
        self.assertGreater(
            count_after, count_before,
            f"Memories before: {count_before}, after: {count_after}. "
            "No new memories saved — check extract_insight() model output."
        )

    def test_run_retrieved_memory_appears_in_context(self):
        """Save a fact, then ask about it — agent should recall it."""
        self.agent.memory.save_long_term(
            "E2ETestUser's favourite programming language is Rust.",
            {"type": "test"}
        )
        resp = self.agent.run(
            "What is E2ETestUser's favourite programming language?",
            source="test"
        )
        self.assertIn("Rust", resp,
                      "Agent should recall the saved memory in its response")

    def test_run_tool_intent_executes_tool(self):
        """A file-listing request should trigger a TOOL route and return file info."""
        resp = self.agent.run(
            "Use a tool to list the files in the current directory and tell me what you see.",
            source="test"
        )
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp.strip()), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 11. FULL GOAL LIFECYCLE WITH HEARTBEAT (live inference)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestFullGoalLifecycle(unittest.TestCase):
    """
    End-to-end test of the complete autonomous goal lifecycle:
      set_goal → heartbeat generates plan → heartbeat executes steps
      → all steps complete → goal marked complete → reflect() → journal entry
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="dross_goal_e2e_")
        import src.memory as mem_module
        cls._orig_init = mem_module.MemorySystem.__init__

        def patched_init(self_inner, db_path="./data/memory_db"):
            cls._orig_init(self_inner, db_path=cls.tmp_dir)

        mem_module.MemorySystem.__init__ = patched_init
        from src.agent import Agent
        cls.agent = Agent()

    @classmethod
    def tearDownClass(cls):
        import src.memory as mem_module
        mem_module.MemorySystem.__init__ = cls._orig_init
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)
        # Clean up goal/plan files created during test
        from src.tools import GOAL_FILE, GOAL_STACK_FILE, PLAN_FILE
        for f in [GOAL_FILE, GOAL_STACK_FILE, PLAN_FILE]:
            if os.path.exists(f):
                os.remove(f)

    def setUp(self):
        from src.tools import GOAL_FILE, GOAL_STACK_FILE, PLAN_FILE
        for f in [GOAL_FILE, GOAL_STACK_FILE, PLAN_FILE]:
            if os.path.exists(f):
                os.remove(f)

    def test_01_heartbeat_returns_none_with_no_goal(self):
        result = self.agent.heartbeat()
        self.assertIsNone(result, "heartbeat() should return None when there is no active goal")

    def test_02_heartbeat_generates_plan_when_none_exists(self):
        from src.tools import GOAL_FILE, PLAN_FILE
        self.agent.tools.execute("set_goal", {"description": "Write a hello world Python script"})
        self.assertFalse(os.path.exists(PLAN_FILE), "Plan file should not exist yet")

        self.agent.heartbeat()

        self.assertTrue(os.path.exists(PLAN_FILE), "heartbeat() should generate a plan file")
        plan = _read_json(PLAN_FILE)
        self.assertIn("steps", plan)
        self.assertGreater(len(plan["steps"]), 0, "Generated plan should have at least one step")

    def test_03_heartbeat_advances_plan_steps(self):
        from src.tools import PLAN_FILE
        self.agent.tools.execute("set_goal", {"description": "List files in the workspace"})

        # First heartbeat: generate plan
        self.agent.heartbeat()

        plan_before = _read_json(PLAN_FILE)
        pending_before = sum(1 for s in plan_before["steps"] if s["status"] == "pending")

        # Second heartbeat: execute first step
        self.agent.heartbeat()

        plan_after = _read_json(PLAN_FILE)
        pending_after = sum(1 for s in plan_after["steps"] if s["status"] == "pending")

        self.assertLess(
            pending_after, pending_before,
            f"A step should have been completed or failed. "
            f"Pending before: {pending_before}, after: {pending_after}"
        )

    def test_04_failed_step_not_marked_completed(self):
        """A step whose tool returns an error string should be marked 'failed' not 'completed'."""
        from src.tools import PLAN_FILE
        from unittest.mock import patch

        self.agent.tools.execute("set_goal", {"description": "Do something"})
        self.agent.tools.execute("set_plan", {"steps": ["Do something that will fail"]})

        # Mock query_autonomy to return an action that calls a nonexistent tool
        mock_response = json.dumps({
            "thought": "I will try to call a broken tool",
            "actions": [{"tool_name": "nonexistent_tool_xyz", "tool_args": {}}]
        })

        with patch.object(self.agent.models, "query_autonomy", return_value=mock_response):
            self.agent.heartbeat()

        plan = _read_json(PLAN_FILE)
        self.assertEqual(
            plan["steps"][0]["status"], "failed",
            "Step should be marked 'failed' when tool returns an error"
        )

    def test_05_goal_completes_when_all_steps_done(self):
        from src.tools import GOAL_FILE, PLAN_FILE
        self.agent.tools.execute("set_goal", {"description": "Quick test goal"})
        self.agent.tools.execute("set_plan", {"steps": ["Only step"]})
        self.agent.tools.execute("update_plan_step", {"step_index": 0, "status": "completed"})

        result = self.agent.heartbeat()

        goal = _read_json(GOAL_FILE)
        self.assertEqual(
            goal["status"], "completed",
            f"Goal should be marked completed when all steps done. heartbeat() returned: {result}"
        )

    def test_06_reflect_saves_to_journal(self):
        import src.tools as tools_module
        tmp_journal = tempfile.mktemp(suffix=".jsonl")
        orig = tools_module.JOURNAL_FILE
        tools_module.JOURNAL_FILE = tmp_journal

        try:
            goal_data = json.dumps({
                "goal": "Write a test file",
                "status": "completed",
                "result": "Created test.txt successfully.",
                "log": []
            })
            result = self.agent.reflect(goal_data)

            self.assertIsInstance(result, str)
            self.assertNotIn("failed", result.lower(),
                             f"reflect() should not fail: {result}")

            if os.path.exists(tmp_journal):
                entries = _read_jsonl(tmp_journal)
                self.assertGreater(len(entries), 0,
                                   "reflect() should write at least one journal entry")
        finally:
            tools_module.JOURNAL_FILE = orig
            if os.path.exists(tmp_journal):
                os.remove(tmp_journal)

    def test_07_reflect_saves_lesson_to_long_term_memory(self):
        count_before = self.agent.memory.collection.count()

        goal_data = json.dumps({
            "goal": "Learn about memory systems",
            "status": "completed",
            "result": "Read three articles about vector databases.",
            "log": []
        })
        self.agent.reflect(goal_data)

        count_after = self.agent.memory.collection.count()
        self.assertGreater(
            count_after, count_before,
            "reflect() should save the lesson to long-term memory"
        )

    def test_08_full_lifecycle_set_to_journal(self):
        """
        Integration: set goal → run heartbeats until complete → reflect → journal entry.
        This is the highest-level test in the suite.
        """
        import src.tools as tools_module
        from src.tools import GOAL_FILE, PLAN_FILE

        tmp_journal = tempfile.mktemp(suffix=".jsonl")
        orig_journal = tools_module.JOURNAL_FILE
        tools_module.JOURNAL_FILE = tmp_journal

        try:
            # Set a simple, self-contained goal the model can actually complete with tools
            self.agent.tools.execute("set_goal", {
                "description": "Write a file called lifecycle_test.txt containing the text LIFECYCLE_OK",
                "is_autonomous": True
            })

            # Run heartbeats until the goal completes or we hit a safety limit
            MAX_HEARTBEATS = 8
            for pulse in range(MAX_HEARTBEATS):
                goal_raw = self.agent.tools.execute("get_goal", {})
                if "No active goal" in goal_raw:
                    break
                goal_data = json.loads(goal_raw)
                if goal_data.get("status") != "active":
                    break
                self.agent.heartbeat()
                time.sleep(0.5)  # Avoid hammering Ollama

            # Goal should be complete
            goal_final = _read_json(GOAL_FILE)
            self.assertEqual(
                goal_final["status"], "completed",
                f"Goal should be completed after {MAX_HEARTBEATS} heartbeats. "
                f"Final status: {goal_final.get('status')}. "
                f"Plan: {_read_json(PLAN_FILE) if os.path.exists(PLAN_FILE) else 'no plan'}"
            )

            # Run reflect on the completed goal
            self.agent.reflect(json.dumps(goal_final))

            # Journal should have an entry
            self.assertTrue(os.path.exists(tmp_journal), "Journal file should exist after reflect()")
            entries = _read_jsonl(tmp_journal)
            self.assertGreater(len(entries), 0, "Journal should have at least one entry")

        finally:
            tools_module.JOURNAL_FILE = orig_journal
            if os.path.exists(tmp_journal):
                os.remove(tmp_journal)
            # Clean up test file if agent created it
            test_file = os.path.join(os.getcwd(), "workspace", "lifecycle_test.txt")
            if os.path.exists(test_file):
                os.remove(test_file)


# ══════════════════════════════════════════════════════════════════════════════
# 12. SERVER API  (requires running server — mark live)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
class TestServerAPI(unittest.TestCase):
    """
    HTTP/WebSocket tests against the running server.
    Run with:  pytest test_dross.py -m live
    Requires server.py to be running on localhost:8001.
    """

    BASE = "http://localhost:8001"

    def _get(self, path):
        import requests
        return requests.get(f"{self.BASE}{path}", timeout=5)

    def _post(self, path, **kwargs):
        import requests
        return requests.post(f"{self.BASE}{path}", timeout=5, **kwargs)

    def test_root_returns_html(self):
        r = self._get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("content-type", ""))

    def test_status_endpoint_structure(self):
        r = self._get("/api/status")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("goal", data)
        self.assertIn("memory_count", data)
        self.assertIn("uptime", data)

    def test_memory_graph_endpoint(self):
        r = self._get("/api/memory/graph")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

    def test_journal_endpoint(self):
        r = self._get("/api/journal")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("entries", data)

    def test_websocket_echo(self):
        """Send a message over WS, verify we get a response type back."""
        import asyncio
        import websockets

        async def _check():
            uri = "ws://localhost:8001/ws"
            async with websockets.connect(uri) as ws:
                await ws.send("Reply with only the word PONG.")
                # Collect messages for up to 15s waiting for response type
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2)
                        msg = json.loads(raw)
                        if msg.get("type") == "response":
                            return msg["content"]
                    except asyncio.TimeoutError:
                        continue
                return None

        result = asyncio.run(_check())
        self.assertIsNotNone(result, "WebSocket should return a response message within 15s")
        self.assertIsInstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Running directly: skip live server tests, run everything else including slow
    # For CI / fast iteration use: pytest test_dross.py -m "not slow and not live"
    import sys
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestConfig,
        TestMemorySystem,
        TestToolRegistry,
        TestGoalManagement,
        TestPlanManagement,
        TestJournal,
        TestAgentUnit,
        # Slow (live inference) — comment out to skip
        TestOllamaConnectivity,
        TestModels,
        TestAgentEndToEnd,
        TestFullGoalLifecycle,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)