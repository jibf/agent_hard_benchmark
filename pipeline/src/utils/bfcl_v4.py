import re
from typing import Tuple


def normalize_bfcl_v4_task_info(task_name: str, question_id: str) -> Tuple[str, str]:
    """Normalize BFCL V4 task name and question id to the canonical format used by human labels.

    - task_name: always "web_search" or "memory"
    - question_id: use the dataset id number (e.g., web_search_89 -> 89, memory_153-notetaker-22 -> 153)
    """
    task_name = task_name or ""
    question_id = question_id or ""

    base_task = "web_search" if "web_search" in task_name else "memory" if "memory" in task_name else task_name

    canonical_id = question_id
    if "memory" in base_task:
        # capture first numeric segment after memory* prefix
        mem_match = re.match(r"^memory(?:_(?:vector|kv|rec_sum))?_(\d+)-", question_id)
        if mem_match:
            canonical_id = mem_match.group(1)
        else:
            # fallback to last digits
            trailing = re.search(r"(\d+)$", question_id)
            canonical_id = trailing.group(1) if trailing else question_id
    elif "web_search" in base_task:
        web_match = re.match(r"^web_search(?:_(?:no_snippet|base))?_(\d+)$", question_id)
        if web_match:
            canonical_id = web_match.group(1)
        else:
            trailing = re.search(r"(\d+)$", question_id)
            canonical_id = trailing.group(1) if trailing else question_id

    return base_task, canonical_id
