import json
import re
import ast
from typing import Optional, Any

def clean_output(text: str) -> str:
    """Removes <think> tags, \\boxed{} wrappers, and other artifacts."""
    if not text:
        return ""
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')

    # Remove \boxed{...} wrappers (common in reasoning models)
    text = re.sub(r'\\boxed\{(.*?)\}', r'\1', text, flags=re.DOTALL)

    return text.strip()

def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers the model sometimes adds."""
    text = text.strip()
    # Handle ```json ... ``` or ``` ... ```
    match = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def extract_json(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from model output."""
    if not text:
        return None

    cleaned = clean_output(text)

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try stripping fences
    cleaned = strip_json_fences(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks if still there
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]

    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Last resort: find first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        snippet = cleaned[start:end+1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # Model sometimes outputs Python-style single-quoted dicts
            try:
                return ast.literal_eval(snippet)
            except Exception:
                pass

    return None
