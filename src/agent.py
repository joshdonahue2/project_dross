import json
import os
import re
from typing import List, Dict, Any, Optional
from .models import ModelManager
from .memory import MemorySystem
from .tools import registry as tool_registry
from .logger import get_logger
from .utils import clean_output, extract_json

logger = get_logger("agent")

try:
    from src.config import MAX_AUTONOMY_STEPS
except ImportError:
    MAX_AUTONOMY_STEPS = 5

class Agent:
    def __init__(self):
        logger.info("Initializing Agent Components...")
        self.models = ModelManager()
        self.memory = MemorySystem()
        self.tools = tool_registry
        logger.info("Agent Ready.")

    def _clean_output(self, text: str) -> str:
        """Removes <think> tags, \\boxed{} wrappers, and other artifacts."""
        return clean_output(text)

    def run(self, user_input: str, source: str = "cli") -> str:
        # 1. Retrieve Memory
        long_term_context = self.memory.retrieve_relevant(user_input)
        short_term_history = self.memory.get_short_term()
        
        # 2. Determine Intent
        intent = self.models.route_request(user_input)
        reasoning = ""
        tool_result = ""

        # Environmental Context â€” only pay the list_files I/O cost for non-trivial intents
        if intent in ("REASON", "TOOL"):
            cwd = os.getcwd()
            os_info = os.name
            local_files = self.tools.execute("list_files", {"path": "."})
            env_context = f"LOCAL ENVIRONMENT:\n- OS: {os_info}\n- CWD: {cwd}\n- FILES: {local_files}\n"
        else:
            env_context = ""

        if intent == "REASON":
            context_str = f"{env_context}\nLong-term Memory: {long_term_context}\nShort-term History: {short_term_history}"
            reasoning = self.models.query_reasoning(user_input, context_str)
        
        elif intent == "TOOL":
            tool_schema = self.tools.get_schemas_str()
            tool_decision_json = self.models.query_tool(f"{env_context}\n{user_input}", tool_schema)

            if tool_decision_json:
                tool_data = self._extract_json(tool_decision_json)
                if tool_data:
                    tool_name = tool_data.get("tool_name") or tool_data.get("name")
                    # Standardize on tool_args; also accept legacy field names
                    tool_args = tool_data.get("tool_args") or tool_data.get("arguments") or tool_data.get("args") or {}
                    if tool_name:
                        tool_result = self.tools.execute(tool_name, tool_args)

        # 3. Build structured context for synthesis
        context_parts = []
        if reasoning:
            context_parts.append(f"[Reasoning Analysis]\n{reasoning}")
        if tool_result:
            context_parts.append(f"[Tool Result]\n{tool_result}")
        if long_term_context:
            context_parts.append(f"[Relevant Memories]\n{long_term_context}")
        
        final_context = (f"{env_context}\n\n" if env_context else "") + "\n\n".join(context_parts)

        response = self.models.query_general(user_input, context=final_context, history=short_term_history)
        
        # Clean the response
        cleaned_response = self._clean_output(response)

        # 4. Update Memory with source tracking
        self.memory.add_short_term("user", user_input, source=source)
        self.memory.add_short_term("assistant", cleaned_response, source=source)
        
        # 5. Auto-Learning â€” only run for substantive exchanges to avoid expensive
        #    LLM calls on trivial one-liners (threshold configurable via AUTO_LEARN_MIN_LENGTH)
        try:
            from src.config import AUTO_LEARN_MIN_LENGTH
        except ImportError:
            AUTO_LEARN_MIN_LENGTH = 200

        if len(user_input) + len(cleaned_response) >= AUTO_LEARN_MIN_LENGTH:
            try:
                interaction_text = f"User: {user_input}\nAssistant: {cleaned_response}"
                insight_data = self.models.extract_insight(interaction_text)

                new_facts = insight_data.get("facts", [])
                relationships = insight_data.get("relationships", [])

                fact_id_map = {}
                # Normalise facts: model sometimes returns dicts like {"name": "Alice"} instead of strings.
                # Flatten any dict into a readable "key: value" sentence as a fallback.
                for raw_fact in new_facts:
                    if isinstance(raw_fact, dict):
                        # Convert {"name": "Alice", "city": "Paris"} â†’ "name: Alice, city: Paris"
                        fact = ", ".join(f"{k}: {v}" for k, v in raw_fact.items())
                    elif isinstance(raw_fact, str):
                        fact = raw_fact
                    else:
                        continue

                    if fact.strip():
                        fid = self.memory.save_long_term(fact, {"type": "auto_learned"})
                        fact_id_map[fact] = fid
                        # Also index by position so integer-based relationship refs still resolve
                        fact_id_map[len(fact_id_map) - 1] = fid
                        print(f"Auto-Learning Insight: {fact}")

                for rel in relationships:
                    rel_source = rel.get("source")
                    rel_target = rel.get("target")
                    rtype = rel.get("type", "references")

                    # Handle both string keys and integer indices
                    sid = fact_id_map.get(rel_source)
                    tid = fact_id_map.get(rel_target)

                    if sid and tid:
                        self.memory.save_relationship(sid, tid, rtype)
            except Exception as e:
                print(f"Auto-learning failed: {e}")
            
        # 6. Episodic Memory Pruning
        try:
            pruned_chunk = self.memory.prune_short_term()
            if pruned_chunk:
                print("Pruning memory and saving episode...")
                summary = self.models.summarize_memory(pruned_chunk)
                # Reject trivial summaries by minimum length (< 40 chars is almost certainly useless)
                if summary and len(summary.strip()) >= 40:
                    self.memory.save_long_term(summary, {"type": "episodic"})
                    print(f"Episodic Memory Saved: {summary}")
                elif summary:
                    print(f"Skipping trivial episodic summary: {summary[:60]!r}")
        except Exception as e:
            print(f"Episodic memory failed: {e}")

        return cleaned_response

    def learn(self, user_input: str, assistant_response: str, feedback: str):
        """
        Explicit learning step. Saves the interaction to long-term memory with positive reinforcement.
        """
        content = f"User: {user_input}\nAssistant: {assistant_response}\nUser Feedback: {feedback}"
        self.memory.save_long_term(content, {"type": "feedback_learning", "rating": "positive"})
        return "Insight saved to long-term memory."

    def _extract_json(self, text: str) -> dict:
        """Robustly extract a JSON object from model output."""
        return extract_json(text)

    def _save_atomic_memories(self, facts: List):
        """Saves a list of facts as individual, high-signal memories."""
        if not facts or not isinstance(facts, list):
            return

        for raw_fact in facts:
            if isinstance(raw_fact, dict):
                # Convert {"name": "Alice", "city": "Paris"} → "name: Alice, city: Paris"
                # Dict-derived facts bypass the min-length filter — they are already structured.
                fact = ", ".join(f"{k}: {v}" for k, v in raw_fact.items())
                if not fact.strip():
                    continue
            elif isinstance(raw_fact, str):
                fact = raw_fact
                if not fact.strip() or len(fact.strip()) < 15:
                    continue
            else:
                continue

            print(f"[Memory] Saving atomic fact: {fact}")
            self.memory.save_long_term(fact.strip(), {"type": "atomic_fact"})

    def reflect(self, goal_data: str) -> str:
        """
        Post-goal reflection. Analyzes what happened, saves lessons,
        and optionally creates new tools.
        """
        try:
            print("[Reflect] Analyzing completed goal...")
            reflection_raw = self.models.query_reflection(goal_data)
            reflection = self._extract_json(reflection_raw)
            
            if not reflection:
                # Save raw text as lesson if it's substantial enough
                if len(reflection_raw.strip()) < 40:
                    print("[Reflect] Skipping trivial reflection.")
                    return "Skipped trivial reflection."
                    
                self.memory.save_long_term(
                    f"Reflection (unparsed): {reflection_raw[:500]}",
                    {"type": "self_improvement"}
                )
                self.tools.execute("write_journal", {"entry": f"Reflection: {reflection_raw[:300]}"})
                return f"Reflection saved (raw): {reflection_raw[:200]}"
            
            # 1. Save Atomic Facts
            atomic_facts = reflection.get("key_facts", [])
            self._save_atomic_memories(atomic_facts)

            # 2. Save structured lesson to memory
            lesson = reflection.get("lessons", "No specific lesson.")
            outcome = reflection.get("outcome", "unknown")
            what_worked = reflection.get("what_worked", "")
            what_failed = reflection.get("what_failed", "")

            lesson_text = (
                f"LESSON [{outcome}]: {lesson}. "
                f"Worked: {what_worked}. Gaps: {what_failed}."
            )

            # Skip saving if the lesson is suspiciously short (< 30 chars = almost certainly noise)
            if len(lesson.strip()) < 30:
                print(f"[Reflect] Skipping trivial lesson: {lesson!r}")
                return "Skipped trivial lesson."
            
            self.memory.save_long_term(lesson_text, {"type": "self_improvement", "outcome": outcome})
            
            # 3. Journal entry
            import json as _json
            self.tools.execute("write_journal", {
                "entry": _json.dumps(reflection)
            })
            
            # 4. Auto-create tool if suggested
            suggested = reflection.get("suggested_tool")
            if suggested and isinstance(suggested, dict) and suggested.get("code"):
                print(f"[Reflect] Creating suggested tool: {suggested.get('name')}")
                result = self.tools.execute("create_tool", {
                    "name": suggested["name"],
                    "description": suggested.get("description", "Auto-created tool"),
                    "code": suggested["code"]
                })
                lesson_text += f" | Auto-created tool: {result}"
            
            print(f"[Reflect] {lesson_text}")
            return lesson_text
            
        except Exception as e:
            print(f"Reflection Error: {e}")
            return f"Reflection failed: {e}"

    def heartbeat(self) -> str:
        """
        Autonomous execution loop. Multi-step: follows a plan if available.
        """
        tool_schemas = self.tools.get_schemas_str()
        all_results = []
        
        # 1. Check Goal
        goal_json = self.tools.execute("get_goal", {})
        if "No active goal" in goal_json or "Error" in goal_json:
             # (Boredom logic skipped here for brevity, keeping same structure)
             return None
        
        goal_data = json.loads(goal_json)
        goal_desc = goal_data.get("goal", "")
        
        # 2. Check Plan
        plan_json = self.tools.execute("get_plan", {})
        if "No plan defined" in plan_json:
            logger.info(f"[Heartbeat] No plan for goal '{goal_desc[:30]}'. Generating...")
            steps = self.models.generate_plan(goal_desc)
            self.tools.execute("set_plan", {"steps": steps})
            plan_json = self.tools.execute("get_plan", {})
        
        plan_data = json.loads(plan_json)
        steps = plan_data.get("steps", [])
        
        # Find next pending step
        next_step_idx = -1
        next_step_desc = ""
        for i, s in enumerate(steps):
            if s["status"] == "pending":
                next_step_idx = i
                next_step_desc = s["description"]
                break
        
        if next_step_idx == -1:
            logger.info("[Heartbeat] Plan finished. Marking goal complete.")
            return self.tools.execute("complete_goal", {"result_summary": "All plan steps completed."})

        # 3. Execute next step
        logger.info(f"[Heartbeat] Executing Step {next_step_idx}: {next_step_desc}")
        
        try:
            # Inject Environmental Context to prevent "Generic AI" personality
            cwd = os.getcwd()
            os_info = os.name
            local_files = self.tools.execute("list_files", {"path": "."})
            
            env_context = f"ENVIRONMENT:\n- OS: {os_info}\n- CWD: {cwd}\n- FILES: {local_files}\n"
            
            # We use query_autonomy but focus it on the current step
            context = f"{env_context}\nCURRENT GOAL: {goal_desc}\nACTIVE STEP ({next_step_idx}): {next_step_desc}\nPLAN STATUS: {plan_json}"
            response = self.models.query_autonomy(
                goal_state=context,
                tools_schema=tool_schemas
            )
            
            data = self._extract_json(response)
            if data and data.get("actions"):
                step_failed = False
                for action in data["actions"]:
                    tool = action.get("tool_name")
                    args = action.get("tool_args", {})
                    if tool:
                        output = self.tools.execute(tool, args)
                        all_results.append(f"{tool} -> {output[:100]}")
                        # If the tool reported an error, mark the step as failed instead of completed
                        if output.lower().startswith("error") or "exception" in output.lower():
                            step_failed = True

                new_status = "failed" if step_failed else "completed"
                self.tools.execute("update_plan_step", {"step_index": next_step_idx, "status": new_status})
                if step_failed:
                    logger.warning(f"[Heartbeat] Step {next_step_idx} marked as failed due to tool error.")
            
        except Exception as e:
            logger.error(f"Heartbeat Error: {e}")
            all_results.append(f"Error: {e}")
        
        return " | ".join(all_results) if all_results else "No actions taken."

    def full_reset(self):
        """Wipes all state EXCEPT for high-level logs and journal."""
        print("[Agent] Initiating FULL RESET...")
        
        # 1. Wipe Memory System (wipe_memory resets the collection AND clears short-term)
        self.memory.wipe_memory()
        self.memory.clear_short_term()  # Belt-and-suspenders: ensure in-memory list is also emptied
        
        # 2. Wipe Goal Files
        from src.tools import GOAL_FILE, GOAL_STACK_FILE
        data_dir = os.path.dirname(GOAL_FILE)
        
        for f in [GOAL_FILE, GOAL_STACK_FILE, os.path.join(data_dir, "subtasks.json")]:
            if os.path.exists(f):
                try:
                    with open(f, 'w', encoding='utf-8') as file:
                        if f.endswith('.json'):
                            file.write("{}")
                        else:
                            file.write("")
                except Exception as e:
                    print(f"Error reset file {f}: {e}")
        
        print("[Agent] Reset Complete. Slate is clean.")
        return "System reset successful. Memory and goals cleared."