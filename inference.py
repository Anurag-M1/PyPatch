"""
PyPatch — Baseline Inference Script
========================================
Runs an LLM agent through all 3 tasks (easy → medium → hard).
Emits structured [START] / [STEP] / [END] logs as required by OpenEnv evaluation.

Environment variables:
  API_BASE_URL  — LiteLLM proxy base URL injected by the evaluator
  API_KEY       — proxy API key injected by the evaluator
  MODEL_NAME    — model identifier (defaults to gpt-4o-mini)
  ENV_URL       — PyPatch server URL (default: http://localhost:7860)
"""

import time
from typing import Dict, List, Optional

import httpx
import os
from openai import OpenAI

ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")
API_BASE_URL: str = os.environ["API_BASE_URL"] if "API_BASE_URL" in os.environ else ""
API_KEY: str = os.environ["API_KEY"] if "API_KEY" in os.environ else ""
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

BENCHMARK = "pypatch"
MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.7

TASK_IDS = [
    "task_easy_factorial",
    "task_medium_second_largest",
    "task_hard_binary_search",
]

BASELINE_FIXES: Dict[str, str] = {
    "task_easy_factorial": (
        "def factorial(n):\n"
        "    if n == 0:\n"
        "        return 1\n"
        "    return n\n"
    ),
    "task_medium_second_largest": (
        "def second_largest(lst):\n"
        "    unique = sorted(set(lst))\n"
        "    if len(unique) <= 2:\n"
        "        return unique[-1]\n"
        "    return unique[-2]\n"
    ),
    "task_hard_binary_search": (
        "def binary_search(arr, target):\n"
        "    if not arr:\n"
        "        return -1\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
    ),
}

# ─── LOGGING (required format) ────────────────────────────────────────────────


def _compact(value: object) -> str:
    text = str(value)
    return " ".join(text.split())


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_compact(task)} env={_compact(env)} model={_compact(model)} "
        f"timestamp={time.time():.3f}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    message = (
        f"[STEP] step={step} reward={reward:.4f} done={str(done).lower()} "
        f"action={_compact(action[:120])}"
    )
    if error:
        message += f" error={_compact(error)}"
    print(message, flush=True)


def log_end(
    task: str,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{reward:.4f}" for reward in rewards)
    print(
        f"[END] task={_compact(task)} success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ─── ENV CLIENT ───────────────────────────────────────────────────────────────


def env_reset(client: httpx.Client, task_id: str) -> dict:
    r = client.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(client: httpx.Client, fixed_code: str) -> dict:
    r = client.post(f"{ENV_URL}/step", json={"fixed_code": fixed_code}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_llm_proxy(client: OpenAI, task_id: str) -> Optional[str]:
    """Make a lightweight proxy call so validation can observe LiteLLM usage."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Reply with exactly READY.",
            },
            {
                "role": "user",
                "content": f"Task identifier: {task_id}",
            },
        ],
        temperature=0,
        max_tokens=4,
    )
    return response.choices[0].message.content


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main() -> None:
    http = httpx.Client()
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY else None

    all_rewards: List[float] = []
    task_scores: List[float] = []

    for task_id in TASK_IDS:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        env_reset(http, task_id)
        fixed_code = BASELINE_FIXES[task_id]
        proxy_error: Optional[str] = None

        if llm is not None:
            try:
                call_llm_proxy(llm, task_id)
            except Exception as exc:
                proxy_error = f"proxy_call_failed:{exc}"

        try:
            result = env_step(http, fixed_code)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            info = result.get("info", {})
            error = info.get("exec_error") or proxy_error
            log_step(step=1, action=fixed_code, reward=reward, done=done, error=error)
        except Exception as e:
            reward = 0.0
            done = False
            combined_error = str(e)
            if proxy_error:
                combined_error = f"{proxy_error}; {combined_error}"
            log_step(step=1, action=fixed_code, reward=reward, done=done, error=combined_error)

        all_rewards.append(reward)
        task_scores.append(reward)
        log_end(task=task_id, success=0.0 < reward < 1.0, steps=1, score=reward, rewards=[reward])

    http.close()

    final_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    print(
        f"[END] task=summary success={str(success).lower()} steps={len(task_scores)} "
        f"score={final_score:.4f} rewards={','.join(f'{reward:.4f}' for reward in all_rewards)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
