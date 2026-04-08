"""
BugFixBench — Baseline Inference Script
========================================
Runs an LLM agent through all 3 tasks (easy → medium → hard).
Emits structured [START] / [STEP] / [END] logs as required by OpenEnv evaluation.

Environment variables:
  API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — API key / Hugging Face token
  ENV_URL       — BugFixBench server URL (default: http://localhost:7860)
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ─── CONFIG ──────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.environ.get("LOCAL_IMAGE_NAME")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

API_KEY: str = HF_TOKEN or ""

BENCHMARK = "bugfixbench"
MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.7

TASK_IDS = [
    "task_easy_factorial",
    "task_medium_second_largest",
    "task_hard_binary_search",
]

# ─── LOGGING (required format) ────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    entry = {
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }
    print(json.dumps(entry), flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    entry = {
        "type": "STEP",
        "step": step,
        "action": action[:500],  # truncate for log readability
        "reward": reward,
        "done": done,
    }
    if error:
        entry["error"] = error
    print(json.dumps(entry), flush=True)


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    entry = {
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }
    print(json.dumps(entry), flush=True)


# ─── ENV CLIENT ───────────────────────────────────────────────────────────────


def env_reset(client: httpx.Client, task_id: str) -> dict:
    r = client.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(client: httpx.Client, fixed_code: str) -> dict:
    r = client.post(f"{ENV_URL}/step", json={"fixed_code": fixed_code}, timeout=30)
    r.raise_for_status()
    return r.json()


# ─── AGENT ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python debugger. 
When given buggy Python code you return ONLY the corrected Python code.
No explanations, no markdown, no triple backticks — just raw, runnable Python code.
Preserve the exact function name and signature."""


def build_user_prompt(
    task_description: str,
    buggy_code: str,
    error_hint: str,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    lines = [
        f"TASK:\n{task_description}",
        f"\nBUGGY CODE:\n{buggy_code}",
        f"\nHINT: {error_hint}",
    ]
    if step > 1:
        lines.append(f"\nYour last attempt scored {last_reward:.2f} (fraction of tests passed).")
        if history:
            lines.append("Recent attempt summary:\n" + "\n".join(history[-2:]))
    lines.append("\nReturn ONLY the corrected Python code:")
    return "\n".join(lines)


def get_fixed_code(
    llm: OpenAI,
    task_description: str,
    buggy_code: str,
    error_hint: str,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        task_description, buggy_code, error_hint, step, last_reward, history
    )
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    code = response.choices[0].message.content.strip()
    # Strip accidental markdown fences
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return code


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "none")
    http = httpx.Client()

    all_rewards: List[float] = []
    task_scores: List[float] = []
    global_step = 0

    log_start(task="all_tasks", env=BENCHMARK, model=MODEL_NAME)

    for task_id in TASK_IDS:
        print(f"[DEBUG] Starting task: {task_id}", flush=True)

        result = env_reset(http, task_id)
        obs = result["observation"]
        history: List[str] = []
        last_reward = 0.0
        task_best_reward = 0.0

        for attempt in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            try:
                fixed_code = get_fixed_code(
                    llm,
                    obs["task_description"],
                    obs["buggy_code"],
                    obs["error_hint"],
                    attempt,
                    last_reward,
                    history,
                )
            except Exception as e:
                log_step(step=global_step + attempt, action="", reward=0.0, done=False, error=str(e))
                continue

            result = env_step(http, fixed_code)
            reward: float = result.get("reward", 0.0)
            done: bool = result.get("done", False)
            info = result.get("info", {})

            last_reward = reward
            task_best_reward = max(task_best_reward, reward)
            all_rewards.append(reward)

            log_step(
                step=global_step + attempt,
                action=fixed_code,
                reward=reward,
                done=done,
                error=info.get("exec_error"),
            )

            history.append(
                f"Attempt {attempt}: {info.get('tests_passed', '?')}/{info.get('tests_total', '?')} tests passed (reward={reward:.2f})"
            )

            obs = result.get("observation") or obs

            if done:
                break

        global_step += attempt  # type: ignore[possibly-undefined]
        task_scores.append(task_best_reward)
        print(f"[DEBUG] Task {task_id} finished. Best reward: {task_best_reward:.2f}", flush=True)

    http.close()

    final_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=global_step, score=final_score, rewards=all_rewards)


if __name__ == "__main__":
    main()
