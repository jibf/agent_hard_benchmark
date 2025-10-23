from typing import Any, Callable, List, Dict, Tuple

from os import getenv

import statistics

import time

import json

import copy

from traceback import format_exc, print_exc

from functools import wraps

from concurrent.futures import ThreadPoolExecutor, as_completed

from tabulate import tabulate

from tqdm import tqdm


DEFAULT_MAX_EXECUTION_RETRIES = 1
EXECUTION_RETRY_DELAY = 1


def handle_exceptions(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_execution_retries = getenv("MAX_EXECUTION_RETRIES")
        if max_execution_retries is not None:
            max_execution_retries = json.loads(max_execution_retries)
        else:
            max_execution_retries = DEFAULT_MAX_EXECUTION_RETRIES

        debug = json.loads(getenv("DEBUG", "false"))

        retry_delay = EXECUTION_RETRY_DELAY
        for attempt in range(max_execution_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e_str = format_exc() if debug else str(e)
                if attempt < max_execution_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed: {e_str}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached. Last error: {e_str}")
                    # pylint: disable=raise-missing-from
                    raise ValueError("Max Retries Reached")

    return wrapper


def parallelize():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_workers = json.loads(getenv("NUM_SAMPLES_PARALLEL"))
            debug = json.loads(getenv("DEBUG"))

            futures = []
            total = len(args[1])
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for arg in args[1]:
                    future = executor.submit(func, arg)
                    futures.append(future)

                results = []
                with tqdm(total=total, desc=f"Processing {func.__name__}") as pbar:
                    for future, arg in zip(as_completed(futures), args[1]):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            if debug:
                                print_exc()
                            else:
                                print(e)
                            results.append(e)
                        finally:
                            pbar.update(1)
            return results

        return wrapper

    return decorator


def print_benchmark_results(accuracies: List[Tuple[str, Dict[str, float], Any, float]]):
    table_data = []
    all_metrics = set()
    for _, metrics, _, _ in accuracies:
        all_metrics.update(metrics.keys())

    # Sort metrics to ensure consistent order
    sorted_metrics = sorted(all_metrics)

    # Prepare headers
    headers = ["Benchmark"] + sorted_metrics + ["Time taken (s)"]

    # Populate table data
    for name, metrics, _, time_taken_s in accuracies:
        row = [name.__name__]
        for metric in sorted_metrics:
            value = metrics.get(metric, "N/A")
            row.append(f"{value:.2%}" if isinstance(value, float) else value)
        row.append(f"{time_taken_s:.0f}")
        table_data.append(row)

    # Calculate and add average row
    avg_row = ["Average"]
    for metric in sorted_metrics:
        values = [
            metrics[metric] for _, metrics, _, _ in accuracies if metric in metrics
        ]
        if values:
            avg_value = statistics.mean(values)
            avg_row.append(f"{avg_value:.2%}")
        else:
            avg_row.append("N/A")

        times_taken_s = [
            time_taken_s for _, _, _, time_taken_s in accuracies if metric in metrics
        ]
        avg_time_taken_s = f"{statistics.mean(times_taken_s):.0f}"
        avg_row.append(avg_time_taken_s)

    table_data.append(avg_row)

    # Print the results in a pretty table
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def as_dict(x):
        if hasattr(x, "model_dump"):
            try: return x.model_dump()
            except Exception: pass
        if hasattr(x, "to_dict"):
            try: return x.to_dict()
            except Exception: pass
        if isinstance(x, dict):
            return dict(x)
        
        # duck-type minimal ChatCompletionMessage
        role = getattr(x, "role", None)
        content = getattr(x, "content", None)
        if role is not None and content is not None:
            d = {"role": role, "content": content}
            name = getattr(x, "name", None)
            if name is not None: d["name"] = name
            tcs = getattr(x, "tool_calls", None)
            if tcs is not None: d["tool_calls"] = [as_dict(tc) for tc in (tcs or [])]
            return d
        return x

def _prune_for_gemini_schema(node):
    """
    Recursively remove JSON Schema fields Gemini's function params don't accept,
    notably 'additionalProperties'. Keep a small, safe subset.
    """
    if isinstance(node, dict):
        t = node.get("type")
        # Always drop additionalProperties anywhere
        node.pop("additionalProperties", None)

        # Whitelist per type
        if t == "object":
            # keep only known-safe keys; ensure properties exists
            allowed = {"type", "properties", "required", "description", "enum"}
            for k in list(node.keys()):
                if k not in allowed:
                    node.pop(k, None)
            node.setdefault("properties", {})
            # recurse into properties
            props = node.get("properties", {})
            for k, v in list(props.items()):
                props[k] = _prune_for_gemini_schema(v)

        elif t == "array":
            allowed = {"type", "items", "description"}
            for k in list(node.keys()):
                if k not in allowed:
                    node.pop(k, None)
            if "items" in node:
                node["items"] = _prune_for_gemini_schema(node["items"])

        else:
            # primitives or missing/unknown type: keep minimal keys
            allowed = {"type", "description", "enum"}
            for k in list(node.keys()):
                if k not in allowed:
                    node.pop(k, None)
            # if no type, coerce to object with empty props (safest)
            if "type" not in node:
                node["type"] = "object"
                node["properties"] = {}

    elif isinstance(node, list):
        return [_prune_for_gemini_schema(x) for x in node]
    return node

def _first_type(t):
    if isinstance(t, list) and t:
        # prefer scalars; fall back to first
        for cand in ("string","number","integer","boolean","null","object","array"):
            if cand in t:
                return cand
        return t[0]
    return t

def gemini_flatten_parameters(params: dict | None, max_props: int = 16) -> dict:
    """
    Make a *shallow* schema Gemini proxies accept:
      - keep only top-level properties
      - coerce objects/arrays/unknown to string
      - keep requireds that still exist
    """
    ALLOWED_SCALAR_TYPES = {"string", "number", "integer", "boolean", "null"}

    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "required": []}

    props = params.get("properties", {})
    if not isinstance(props, dict):
        props = {}

    new_props = {}
    for i, (name, spec) in enumerate(props.items()):
        if i >= max_props:
            break
        spec = spec if isinstance(spec, dict) else {}
        t = _first_type(spec.get("type"))
        if t in ALLOWED_SCALAR_TYPES:
            new_props[name] = {"type": t}
        else:
            new_props[name] = {"type": "string"}  # collapse nested/unknown types

    required = params.get("required", [])
    if not isinstance(required, list):
        required = []
    required = [r for r in required if r in new_props]

    return {"type": "object", "properties": new_props, "required": required}

def gemini_sanitize_tools(tools):
    """Normalize tool schema for OpenAI-compatible gateways."""
    out = []
    for t in tools or []:
        tc = dict(t) if isinstance(t, dict) else as_dict(t)
        tc["type"] = "function"
        fn = (tc.get("function") or {})
        # prune non-standard fields
        fn.pop("returns", None); fn.pop("strict", None)
        if not isinstance(fn.get("parameters"), dict):
            fn["parameters"] = {"type": "object", "properties": {}, "required": []}
        fn["parameters"] = gemini_flatten_parameters(fn.get("parameters"))
        tc["function"] = fn
        out.append(tc)
    return out

def gemini_normalize_messages(msgs):
    """
    Normalize messages for Gemini-compatible gateways:
    - remove None values
    - ensure content is a string
    - trim keys by role
    - convert 'system' → 'user'
    - wrap role:tool content into a JSON object string {"value": ...}
    """
    import json
    cleaned = []
    for raw in msgs:
        # flatten any Pydantic/OpenAI object to dict
        if hasattr(raw, "model_dump"):
            m = raw.model_dump()
        elif hasattr(raw, "to_dict"):
            m = raw.to_dict()
        else:
            m = dict(raw)

        # drop nulls
        m = {k: v for k, v in m.items() if v is not None}

        role = m.get("role")
        content = m.get("content")

        # ensure string content
        if not isinstance(content, str):
            m["content"] = "" if content is None else (
                str(content) if not isinstance(content, list) else ""
            )

        # role-based sanitization
        if role == "assistant":
            keep = {"role", "content", "tool_calls"}
            m = {k: v for k, v in m.items() if k in keep}
            if not m.get("tool_calls"):
                m.pop("tool_calls", None)

        elif role == "tool":
            # Gemini expects function_response.response to be an object
            keep = {"role", "tool_call_id", "name", "content"}
            m = {k: v for k, v in m.items() if k in keep}

            c = m.get("content")
            try:
                parsed = json.loads(c) if isinstance(c, str) else c
            except Exception:
                parsed = c
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            m["content"] = json.dumps(parsed)

        elif role == "system":
            # Convert to user message to avoid schema errors
            sys_text = m.get("content", "")
            m = {"role": "user", "content": f"[SYSTEM INSTRUCTION]\n{sys_text}"}

        else:
            # user or others
            m = {"role": role, "content": m.get("content", "")}

        cleaned.append(m)
    return cleaned