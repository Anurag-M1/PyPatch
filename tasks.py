import ast
import sys
from typing import Any, Dict, List, Tuple
from io import StringIO


# ─── TASK DEFINITIONS ─────────────────────────────────────────────────────────

TASKS: List[Dict[str, Any]] = [
    {
        "id": "task_easy_factorial",
        "name": "Fix Factorial Function",
        "difficulty": "easy",
        "description": (
            "Fix the syntax errors in this recursive factorial function.\n"
            "The function should return n! (n factorial) for any non-negative integer n.\n\n"
            "Expected behaviour:\n"
            "  factorial(0)  → 1\n"
            "  factorial(1)  → 1\n"
            "  factorial(5)  → 120\n"
            "  factorial(10) → 3628800"
        ),
        "buggy_code": (
            "def factorial(n)\n"
            "    if n = 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        "error_hint": (
            "Hint: Look carefully at the function signature line and the comparison "
            "operator inside the if statement."
        ),
        "grader": "python:tasks.grade_task",
        "score": 0.32,
        "function_name": "factorial",
        "test_cases": [
            {"args": (0,), "expected": 1},
            {"args": (1,), "expected": 1},
            {"args": (5,), "expected": 120},
            {"args": (10,), "expected": 3628800},
        ],
    },
    {
        "id": "task_medium_second_largest",
        "name": "Fix Second Largest Element",
        "difficulty": "medium",
        "description": (
            "Fix the logic error in this function that should return the second largest "
            "unique element from a list of integers.\n\n"
            "Expected behaviour:\n"
            "  second_largest([1, 2, 3])       → 2\n"
            "  second_largest([5, 1, 3, 2, 4]) → 4\n"
            "  second_largest([10, 10, 9, 8])  → 9\n"
            "  second_largest([100, 50])        → 50\n"
            "  second_largest([-1, -5, -2])    → -2"
        ),
        "buggy_code": (
            "def second_largest(lst):\n"
            "    unique = list(set(lst))\n"
            "    unique.sort()\n"
            "    return unique[-1]  # Bug: returns the largest, not the second largest\n"
        ),
        "error_hint": (
            "Hint: The function sorts and gets the last element. "
            "Think about which index gives the second largest."
        ),
        "grader": "python:tasks.grade_task",
        "score": 0.33,
        "function_name": "second_largest",
        "test_cases": [
            {"args": ([1, 2, 3],), "expected": 2},
            {"args": ([5, 1, 3, 2, 4],), "expected": 4},
            {"args": ([10, 10, 9, 8],), "expected": 9},
            {"args": ([100, 50],), "expected": 50},
            {"args": ([-1, -5, -2],), "expected": -2},
        ],
    },
    {
        "id": "task_hard_binary_search",
        "name": "Fix Binary Search Algorithm",
        "difficulty": "hard",
        "description": (
            "Fix the multiple algorithmic bugs in this binary search implementation.\n"
            "The function should return the index of `target` in the sorted array `arr`, "
            "or -1 if not found.\n\n"
            "Expected behaviour:\n"
            "  binary_search([1, 3, 5, 7, 9], 5) → 2\n"
            "  binary_search([1, 3, 5, 7, 9], 1) → 0\n"
            "  binary_search([1, 3, 5, 7, 9], 9) → 4\n"
            "  binary_search([1, 3, 5, 7, 9], 4) → -1\n"
            "  binary_search([2], 2)              → 0\n"
            "  binary_search([], 1)               → -1\n"
            "  binary_search([1, 2, 3, 4, 5], 3) → 2"
        ),
        "buggy_code": (
            "def binary_search(arr, target):\n"
            "    if not arr:\n"
            "        return -1\n"
            "    left, right = 0, len(arr)      # Bug 1: off-by-one, should be len(arr)-1\n"
            "    while left < right:             # Bug 2: should be left <= right\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid              # Bug 3: infinite loop, should be mid+1\n"
            "        else:\n"
            "            right = mid             # Bug 4: should be mid-1\n"
            "    return -1\n"
        ),
        "error_hint": (
            "Hint: There are 4 bugs — check the initial right boundary, the while condition, "
            "and both pointer update lines."
        ),
        "grader": "python:tasks.grade_task",
        "score": 0.35,
        "function_name": "binary_search",
        "test_cases": [
            {"args": ([1, 3, 5, 7, 9], 5), "expected": 2},
            {"args": ([1, 3, 5, 7, 9], 1), "expected": 0},
            {"args": ([1, 3, 5, 7, 9], 9), "expected": 4},
            {"args": ([1, 3, 5, 7, 9], 4), "expected": -1},
            {"args": ([2], 2), "expected": 0},
            {"args": ([], 1), "expected": -1},
            {"args": ([1, 2, 3, 4, 5], 3), "expected": 2},
        ],
    },
]

TASK_MAP: Dict[str, Dict] = {t["id"]: t for t in TASKS}


# ─── SAFE EXEC HELPERS ────────────────────────────────────────────────────────

_SAFE_BUILTINS = {
    "range": range, "len": len, "list": list, "set": set,
    "sorted": sorted, "min": min, "max": max, "abs": abs,
    "int": int, "float": float, "str": str, "bool": bool,
    "sum": sum, "enumerate": enumerate, "zip": zip,
    "isinstance": isinstance, "type": type, "print": print,
    "True": True, "False": False, "None": None,
}


def _safe_exec(code: str) -> Tuple[bool, dict, str]:
    """
    Parse and exec code in a restricted namespace.
    Returns (success, namespace, error_message).
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, {}, f"SyntaxError: {e}"

    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(code, namespace)  # noqa: S102
        return True, namespace, ""
    except Exception as e:
        return False, {}, f"{type(e).__name__}: {e}"


# ─── GRADER ───────────────────────────────────────────────────────────────────

def grade_task(task: Dict[str, Any], fixed_code: str) -> Tuple[float, Dict[str, Any]]:
    """
    Grade the agent's fixed code against the task's test cases.
    Returns (score 0.0–1.0, detailed info dict).
    """
    test_cases = task["test_cases"]
    fn_name = task["function_name"]
    total = len(test_cases)

    ok, namespace, exec_error = _safe_exec(fixed_code)
    if not ok:
        return 0.0, {
            "error": exec_error,
            "passed": 0,
            "total": total,
            "results": [],
        }

    fn = namespace.get(fn_name)
    if fn is None:
        return 0.0, {
            "error": f"Function `{fn_name}` not found in submitted code.",
            "passed": 0,
            "total": total,
            "results": [],
        }

    passed = 0
    results = []
    for tc in test_cases:
        try:
            output = fn(*tc["args"])
            correct = output == tc["expected"]
            if correct:
                passed += 1
            results.append({
                "args": str(tc["args"]),
                "expected": tc["expected"],
                "got": output,
                "passed": correct,
            })
        except Exception as e:
            results.append({
                "args": str(tc["args"]),
                "expected": tc["expected"],
                "got": None,
                "passed": False,
                "error": str(e),
            })

    score = round(passed / total, 4)
    return score, {
        "passed": passed,
        "total": total,
        "score": score,
        "results": results,
        "error": None,
    }
