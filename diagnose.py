"""
DROSS Diagnostic Script
Run from your project root: python diagnose.py
Tests every subsystem independently so you can see exactly what's working.
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path so src imports work
sys.path.insert(0, os.path.abspath("."))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[96m[INFO]\033[0m"

results = []

def check(label, passed, detail=""):
    icon = PASS if passed else FAIL
    print(f"  {icon} {label}")
    if detail:
        print(f"         {detail}")
    results.append((label, passed))

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ─────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────
section("1. CONFIG")
try:
    from src.config import (
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, OLLAMA_HOSTS,
        OLLAMA_NUM_CTX, AUTO_LEARN_MIN_LENGTH
    )
    check("src.config imports OK", True)
    check("TELEGRAM_BOT_TOKEN set", bool(TELEGRAM_BOT_TOKEN),
          f"Value: {'***' + TELEGRAM_BOT_TOKEN[-4:] if TELEGRAM_BOT_TOKEN else 'EMPTY — Telegram will be disabled'}")
    check("TELEGRAM_CHAT_ID set", bool(TELEGRAM_CHAT_ID),
          f"Value: {TELEGRAM_CHAT_ID or 'EMPTY'}")
    check("OLLAMA_HOSTS configured", bool(OLLAMA_HOSTS),
          f"Hosts: {OLLAMA_HOSTS}")
    print(f"  {INFO} AUTO_LEARN_MIN_LENGTH = {AUTO_LEARN_MIN_LENGTH} chars")
    print(f"         (combined user+response must exceed this to trigger insight extraction)")
except Exception as e:
    check("src.config imports OK", False, str(e))

# ─────────────────────────────────────────────────────────
# 2. OLLAMA CONNECTIVITY
# ─────────────────────────────────────────────────────────
section("2. OLLAMA CONNECTIVITY")
try:
    from ollama import Client
    from src.config import OLLAMA_HOSTS
    for host in OLLAMA_HOSTS:
        try:
            c = Client(host=host)
            models = c.list()
            model_names = [m.model for m in models.models]
            check(f"Reachable: {host}", True, f"Models available: {', '.join(model_names[:5])}")
        except Exception as e:
            check(f"Reachable: {host}", False, str(e))
except Exception as e:
    check("Ollama client import", False, str(e))

# ─────────────────────────────────────────────────────────
# 3. CHROMADB / MEMORY
# ─────────────────────────────────────────────────────────
section("3. CHROMADB / MEMORY")
try:
    from src.memory import MemorySystem
    mem = MemorySystem()
    check("MemorySystem initializes", True)

    # Count existing memories
    count = mem.collection.count()
    print(f"  {INFO} Existing memories in DB: {count}")

    # Test save
    test_content = f"DIAGNOSTIC TEST MEMORY — {datetime.now().isoformat()}"
    mem_id = mem.save_long_term(test_content, {"type": "diagnostic"})
    check("save_long_term() works", bool(mem_id), f"Saved with id: {mem_id}")

    # Test retrieve
    result = mem.retrieve_relevant("diagnostic test memory")
    check("retrieve_relevant() returns data", bool(result),
          f"Retrieved: {result[:80] if result else 'NOTHING — this is the core bug if failing'}")

    # Test short term
    mem.add_short_term("user", "hello diagnostic", source="test")
    st = mem.get_short_term()
    check("add_short_term() / get_short_term() works", len(st) > 0)

    # Test clear_short_term exists
    mem.clear_short_term()
    check("clear_short_term() exists and works", len(mem.get_short_term()) == 0)

    # Clean up test memory
    mem.delete_memories_containing("DIAGNOSTIC TEST MEMORY")

except Exception as e:
    check("MemorySystem", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────
# 4. TOOL REGISTRY
# ─────────────────────────────────────────────────────────
section("4. TOOL REGISTRY")
try:
    from src.tools import registry
    tool_names = list(registry.tools.keys())
    check("Tool registry loads", len(tool_names) > 0, f"Tools found: {len(tool_names)}")
    print(f"  {INFO} Registered tools: {', '.join(tool_names)}")

    # Test list_files
    result = registry.execute("list_files", {"path": "."})
    check("list_files() executes", "Error" not in result, f"Result snippet: {result[:60]}")

    # Test schema quality — check that types are JSON Schema strings not Python repr
    schemas = json.loads(registry.get_schemas_str())
    bad_schemas = []
    for s in schemas:
        props = s.get("parameters", {}).get("properties", {})
        for pname, pval in props.items():
            t = pval.get("type", "")
            if "<class" in t or "inspect" in t:
                bad_schemas.append(f"{s['name']}.{pname}: {t}")
    check("Tool schemas use JSON Schema types (not Python repr)", len(bad_schemas) == 0,
          f"Bad schemas: {bad_schemas}" if bad_schemas else "All clean")

    # Test path traversal is blocked
    try:
        from src.tools import _get_safe_path
        _get_safe_path("../../etc/passwd")
        check("Path traversal blocked", False, "Should have raised ValueError!")
    except ValueError as e:
        check("Path traversal blocked", True, str(e)[:80])

except Exception as e:
    check("Tool registry", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────
# 5. GOAL & PLAN FILES
# ─────────────────────────────────────────────────────────
section("5. GOAL & PLAN FILES")
DATA_DIR = os.path.abspath("data")
GOAL_FILE = os.path.join(DATA_DIR, "goal.json")
PLAN_FILE = os.path.join(DATA_DIR, "plan.json")
JOURNAL_FILE = os.path.join(DATA_DIR, "journal.jsonl")

check("data/ directory exists", os.path.exists(DATA_DIR),
      f"Path: {DATA_DIR}")

if os.path.exists(GOAL_FILE):
    try:
        with open(GOAL_FILE) as f:
            goal = json.load(f)
        status = goal.get("status", "unknown")
        desc = goal.get("goal", "")[:60]
        print(f"  {INFO} Current goal: [{status}] {desc}")
        check("goal.json is valid JSON", True)
    except Exception as e:
        check("goal.json is valid JSON", False, str(e))
else:
    print(f"  {INFO} No goal.json — no active goal has been set yet")

if os.path.exists(PLAN_FILE):
    try:
        with open(PLAN_FILE) as f:
            plan = json.load(f)
        steps = plan.get("steps", [])
        print(f"  {INFO} Plan has {len(steps)} step(s):")
        for i, s in enumerate(steps):
            print(f"         [{s.get('status','?')}] Step {i}: {s.get('description','')[:50]}")
        check("plan.json is valid JSON", True)
    except Exception as e:
        check("plan.json is valid JSON", False, str(e))
else:
    print(f"  {INFO} No plan.json — no plan has been generated yet")

# Journal
if os.path.exists(JOURNAL_FILE):
    with open(JOURNAL_FILE) as f:
        lines = [l for l in f.readlines() if l.strip()]
    print(f"  {INFO} Journal has {len(lines)} entries")
    if lines:
        try:
            last = json.loads(lines[-1])
            print(f"  {INFO} Last entry: [{last.get('timestamp','?')}] {str(last.get('entry',''))[:80]}")
        except:
            pass
    check("Journal has entries", len(lines) > 0,
          "Journal is empty — reflect() is only called after goal completion")
else:
    print(f"  {INFO} No journal.jsonl yet — journal is written by reflect() after a goal completes")

# ─────────────────────────────────────────────────────────
# 6. MODELS — LIVE INFERENCE TESTS
# ─────────────────────────────────────────────────────────
section("6. MODELS — LIVE INFERENCE (may take 10-30s)")
try:
    from src.models import ModelManager
    mm = ModelManager()
    check("ModelManager initializes", True)

    # Route request
    print(f"  {INFO} Testing route_request()...")
    route = mm.route_request("list the files in my workspace")
    check("route_request() returns TOOL/REASON/DIRECT", route in ("TOOL", "REASON", "DIRECT"),
          f"Got: '{route}'")

    # General query
    print(f"  {INFO} Testing query_general() (short prompt)...")
    resp = mm.query_general("Reply with only the word PONG.", context="", history=[])
    check("query_general() returns non-empty", bool(resp and resp.strip()),
          f"Response snippet: {resp[:80]}")

    # Extract insight — this is the memory pipeline trigger
    print(f"  {INFO} Testing extract_insight() (this feeds the memory system)...")
    test_interaction = (
        "User: My name is TestUser and I live in Berlin. I work as a software engineer.\n"
        "Assistant: Nice to meet you TestUser! Berlin is a great city for software engineers."
    )
    insight = mm.extract_insight(test_interaction)
    facts = insight.get("facts", [])
    rels = insight.get("relationships", [])
    check("extract_insight() returns facts", len(facts) > 0,
          f"Facts: {facts}" if facts else "No facts extracted — auto-learning will silently do nothing")
    check("extract_insight() returns relationships", isinstance(rels, list),
          f"Relationships: {rels}")

except Exception as e:
    check("ModelManager", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────
# 7. END-TO-END: agent.run() PIPELINE
# ─────────────────────────────────────────────────────────
section("7. END-TO-END: agent.run() PIPELINE")
try:
    from src.agent import Agent
    print(f"  {INFO} Initializing Agent (may take a moment)...")
    agent = Agent()
    check("Agent initializes", True)

    mem_before = agent.memory.collection.count()

    # Use a long enough prompt to exceed AUTO_LEARN_MIN_LENGTH
    test_prompt = (
        "My name is DiagnosticUser and I am running a test of the DROSS memory system right now. "
        "Please acknowledge that you have received this message and confirm you understand who I am."
    )
    print(f"  {INFO} Running agent.run() with {len(test_prompt)}-char prompt...")
    print(f"  {INFO} AUTO_LEARN_MIN_LENGTH threshold: needs combined chars > threshold")
    response = agent.run(test_prompt, source="diagnostic")
    check("agent.run() returns response", bool(response), f"Response: {response[:100]}")

    mem_after = agent.memory.collection.count()
    new_memories = mem_after - mem_before
    check(f"New memories were saved ({new_memories} new)", new_memories > 0,
          f"Before: {mem_before}, After: {mem_after}. "
          f"{'ZERO new memories — check AUTO_LEARN_MIN_LENGTH or extract_insight()' if new_memories == 0 else 'Memory pipeline working!'}")

    # Verify retrieval works on what we just saved
    retrieved = agent.memory.retrieve_relevant("DiagnosticUser memory test")
    check("retrieve_relevant finds newly saved memory", bool(retrieved),
          f"Retrieved: {retrieved[:100] if retrieved else 'NOTHING'}")

except Exception as e:
    check("Agent end-to-end", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────
# 8. GOAL + TASK WORKFLOW TEST
# ─────────────────────────────────────────────────────────
section("8. GOAL + TASK WORKFLOW")
try:
    from src.tools import registry as tr
    result = tr.execute("set_goal", {"description": "DIAGNOSTIC: Test goal system", "is_autonomous": False})
    check("set_goal() works", "Error" not in result, result)

    result = tr.execute("get_goal", {})
    check("get_goal() returns active goal", "DIAGNOSTIC" in result, result[:80])

    result = tr.execute("add_subtask", {"subtask": "Step 1: verify subtasks work"})
    check("add_subtask() works", "Error" not in result, result)

    result = tr.execute("list_subtasks", {})
    check("list_subtasks() shows subtask", "Step 1" in result, result)

    result = tr.execute("set_plan", {"steps": ["Diagnose system", "Verify tools", "Report results"]})
    check("set_plan() works", "Error" not in result, result)

    result = tr.execute("get_plan", {})
    check("get_plan() returns plan", "pending" in result, result[:80])

    result = tr.execute("update_plan_step", {"step_index": 0, "status": "completed"})
    check("update_plan_step() works", "Error" not in result, result)

    result = tr.execute("complete_goal", {"result_summary": "Diagnostic complete."})
    check("complete_goal() works", "Error" not in result, result)

    result = tr.execute("write_journal", {"entry": "Diagnostic test journal entry."})
    check("write_journal() works", "Error" not in result, result)

    if os.path.exists(JOURNAL_FILE):
        check("Journal file was created/updated", True, JOURNAL_FILE)
    else:
        check("Journal file was created/updated", False, f"Not found at {JOURNAL_FILE}")

except Exception as e:
    check("Goal/task workflow", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"  {PASS} {passed} checks passed")
if failed:
    print(f"  {FAIL} {failed} checks FAILED:")
    for label, ok in results:
        if not ok:
            print(f"         - {label}")
else:
    print(f"  All systems nominal.")

print(f"\n{'='*55}")
print("  COMMON CAUSES OF SILENT FAILURES:")
print("='*55")
print("  1. AUTO_LEARN_MIN_LENGTH: short chat exchanges won't")
print("     trigger memory. Try longer, substantive messages.")
print("  2. Journal only writes via reflect() which only fires")
print("     after agent.heartbeat() completes a goal.")
print("     → Set a goal via chat ('set a goal to X') then wait")
print("       for the heartbeat timer to process it.")
print("  3. Knowledge graph needs memories first. Once memories")
print("     exist, switch to the Graph tab to see them rendered.")
print("  4. Tasks/subtasks need an active goal. The agent sets")
print("     these via tools — it won't do it for chitchat.")
print(f"{'='*55}\n")