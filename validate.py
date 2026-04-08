#!/usr/bin/env python3
"""
BugFixBench — Pre-Submission Validator
=======================================
Run this BEFORE submitting to verify all OpenEnv checklist requirements pass.

Usage:
    # With server already running:
    python validate.py

    # Auto-start server:
    python validate.py --start-server
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx

ENV_URL = "http://localhost:7860"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results: List[Tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    print(f"  {status}  {name}", end="")
    if detail:
        print(f"  →  {detail}", end="")
    print()
    results.append((name, passed, detail))
    return passed


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ─── CHECKS ───────────────────────────────────────────────────────────────────


def check_health(c: httpx.Client) -> bool:
    section("1. Health Check  (GET /)")
    try:
        r = c.get(f"{ENV_URL}/", timeout=10)
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        check("GET / returns 200", r.status_code == 200, f"status={r.status_code}")
        check("body contains status=ok", data.get("status") == "ok", str(data))
        check("env field present", "env" in data, str(data.get("env")))
        return ok
    except Exception as e:
        check("Server reachable", False, str(e))
        return False


def check_tasks(c: httpx.Client) -> bool:
    section("2. Task Enumeration  (GET /tasks)")
    try:
        r = c.get(f"{ENV_URL}/tasks", timeout=10)
        data = r.json()
        tasks = data.get("tasks", [])
        check("GET /tasks returns 200", r.status_code == 200)
        check("At least 3 tasks", len(tasks) >= 3, f"found {len(tasks)}")
        difficulties = {t.get("difficulty") for t in tasks}
        check("Covers easy/medium/hard", {"easy", "medium", "hard"} <= difficulties, str(difficulties))
        check("All tasks have id/name/difficulty", all(
            t.get("id") and t.get("name") and t.get("difficulty") for t in tasks
        ))
        return len(tasks) >= 3
    except Exception as e:
        check("Tasks endpoint", False, str(e))
        return False


def check_reset(c: httpx.Client) -> Dict[str, Any]:
    section("3. Reset  (POST /reset)")
    try:
        r = c.post(f"{ENV_URL}/reset", json={}, timeout=10)
        data = r.json()
        obs = data.get("observation", {})
        check("POST /reset returns 200", r.status_code == 200, f"status={r.status_code}")
        check("observation present", "observation" in data)
        check("done field present", "done" in data)
        check("reward field present", "reward" in data)
        for field in ["task_id", "task_name", "difficulty", "task_description", "buggy_code", "error_hint"]:
            check(f"observation.{field} present", field in obs)
        return data
    except Exception as e:
        check("Reset endpoint", False, str(e))
        return {}


def check_step(c: httpx.Client) -> bool:
    section("4. Step  (POST /step)")
    # reset first
    c.post(f"{ENV_URL}/reset", json={"task_id": "task_easy_factorial"}, timeout=10)

    # submit correct fix
    fixed = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n"
    try:
        r = c.post(f"{ENV_URL}/step", json={"fixed_code": fixed}, timeout=15)
        data = r.json()
        reward = data.get("reward", -1)
        check("POST /step returns 200", r.status_code == 200)
        check("reward in response", "reward" in data)
        check("reward is 0.0–1.0", 0.0 <= reward <= 1.0, f"reward={reward}")
        check("done field present", "done" in data)
        check("Correct fix scores 1.0", reward == 1.0, f"reward={reward}")
        return True
    except Exception as e:
        check("Step endpoint", False, str(e))
        return False


def check_partial_reward(c: httpx.Client) -> bool:
    section("5. Partial Reward Signal")
    c.post(f"{ENV_URL}/reset", json={"task_id": "task_hard_binary_search"}, timeout=10)
    # submit buggy code — should give partial credit or 0
    broken = "def binary_search(arr, target):\n    return -1  # lazy agent\n"
    try:
        r = c.post(f"{ENV_URL}/step", json={"fixed_code": broken}, timeout=15)
        data = r.json()
        reward = data.get("reward", -1)
        check("Partial/zero reward for bad fix", 0.0 <= reward < 1.0, f"reward={reward}")
        return True
    except Exception as e:
        check("Partial reward check", False, str(e))
        return False


def check_state(c: httpx.Client) -> bool:
    section("6. State  (GET /state)")
    try:
        r = c.get(f"{ENV_URL}/state", timeout=10)
        data = r.json()
        check("GET /state returns 200", r.status_code == 200)
        for field in ["step_count", "done", "total_reward"]:
            check(f"state.{field} present", field in data)
        return r.status_code == 200
    except Exception as e:
        check("State endpoint", False, str(e))
        return False


def check_reward_graders(c: httpx.Client) -> bool:
    section("7. Per-Task Grader Scores  (all 3 tasks)")

    correct = {
        "task_easy_factorial": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n",
        "task_medium_second_largest": "def second_largest(lst):\n    unique = list(set(lst))\n    unique.sort()\n    return unique[-2]\n",
        "task_hard_binary_search": (
            "def binary_search(arr, target):\n"
            "    if not arr:\n        return -1\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        elif arr[mid] < target:\n            left = mid + 1\n"
            "        else:\n            right = mid - 1\n"
            "    return -1\n"
        ),
    }

    all_passed = True
    for task_id, code in correct.items():
        c.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
        r = c.post(f"{ENV_URL}/step", json={"fixed_code": code}, timeout=15)
        data = r.json()
        reward = data.get("reward", -1)
        passed = reward == 1.0
        if not passed:
            all_passed = False
        check(f"{task_id} scores 1.0", passed, f"reward={reward}")

    return all_passed


def check_openenv_yaml() -> bool:
    section("8. openenv.yaml Compliance")
    import os
    import yaml  # type: ignore[import]

    path = "openenv.yaml"
    if not os.path.exists(path):
        check("openenv.yaml exists", False)
        return False
    check("openenv.yaml exists", True)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        for field in ["name", "version", "description", "tasks", "observation_space", "action_space", "reward"]:
            check(f"yaml.{field} present", field in data)
        tasks = data.get("tasks", [])
        check("3+ tasks in yaml", len(tasks) >= 3, str(len(tasks)))
        return True
    except Exception as e:
        check("yaml parseable", False, str(e))
        return False


def check_inference_script() -> bool:
    section("9. inference.py")
    import os
    exists = os.path.exists("inference.py")
    check("inference.py in root directory", exists)
    if exists:
        with open("inference.py") as f:
            code = f.read()
        check("Uses OpenAI client", "OpenAI(" in code)
        check("API_BASE_URL variable", "API_BASE_URL" in code)
        check("MODEL_NAME variable", "MODEL_NAME" in code)
        check("HF_TOKEN variable", "HF_TOKEN" in code)
        check("[START] log emitted", 'log_start' in code or '"START"' in code)
        check("[STEP] log emitted", 'log_step' in code or '"STEP"' in code)
        check("[END] log emitted", 'log_end' in code or '"END"' in code)
    return exists


def check_dockerfile() -> bool:
    section("10. Dockerfile")
    import os
    exists = os.path.exists("Dockerfile")
    check("Dockerfile exists", exists)
    if exists:
        with open("Dockerfile") as f:
            content = f.read()
        check("Exposes port 7860", "7860" in content)
        check("Has CMD/ENTRYPOINT", "CMD" in content or "ENTRYPOINT" in content)
    return exists


# ─── SUMMARY ──────────────────────────────────────────────────────────────────


def print_summary() -> bool:
    total = len(results)
    passed_n = sum(1 for _, p, _ in results if p)
    failed = [(n, d) for n, p, d in results if not p]

    print(f"\n{'═' * 55}")
    print(f"  VALIDATION SUMMARY: {passed_n}/{total} checks passed")
    print(f"{'═' * 55}")

    if failed:
        print(f"\n  {FAIL} FAILED CHECKS:")
        for name, detail in failed:
            print(f"     • {name}")
            if detail:
                print(f"       {detail}")
    else:
        print(f"\n  {PASS} ALL CHECKS PASSED — safe to submit!")

    score_pct = passed_n / total * 100
    print(f"\n  Validation score: {score_pct:.0f}%")
    print(f"{'═' * 55}\n")
    return len(failed) == 0


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────


def main() -> None:
    global ENV_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-server", action="store_true",
                        help="Auto-start uvicorn before validating")
    parser.add_argument("--url", default=ENV_URL, help="Server URL")
    args = parser.parse_args()

    ENV_URL = args.url

    server_proc = None
    if args.start_server:
        print("Starting server...")
        server_proc = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       BugFixBench — Pre-Submission Validator         ║")
    print("╚══════════════════════════════════════════════════════╝")

    try:
        c = httpx.Client()

        ok = check_health(c)
        if not ok:
            print("\n  ⚠️  Server not reachable. Start it with:")
            print("      uvicorn main:app --host 0.0.0.0 --port 7860")
            print("  Then re-run: python validate.py\n")
            sys.exit(1)

        check_tasks(c)
        check_reset(c)
        check_step(c)
        check_partial_reward(c)
        check_state(c)
        check_reward_graders(c)

        # Try yaml check (needs pyyaml)
        try:
            import yaml  # type: ignore[import]
            check_openenv_yaml()
        except ImportError:
            print("\n  ⚠️  pyyaml not installed — skipping yaml check")
            print("      pip install pyyaml")

        check_inference_script()
        check_dockerfile()

        c.close()
        all_ok = print_summary()
        sys.exit(0 if all_ok else 1)

    finally:
        if server_proc:
            server_proc.terminate()


if __name__ == "__main__":
    main()
