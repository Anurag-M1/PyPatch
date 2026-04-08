---
title: PyPatch
emoji: 🐛
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# PyPatch 🐛

**An OpenEnv-compliant RL environment for AI-powered Python code debugging.**

PyPatch challenges AI agents to identify and fix bugs in Python code across three difficulty levels. Reward is proportional to the fraction of hidden test cases the agent's fix passes — giving rich, continuous feedback for reinforcement learning.

---

## Environment Overview

| Property | Value |
|---|---|
| **Tasks** | 3 (easy → medium → hard) |
| **Max steps per task** | 5 |
| **Reward range** | 0.0 – 1.0 |
| **Termination** | All tests pass OR max steps reached |
| **API** | OpenEnv-compliant REST (FastAPI) |

---

## Tasks

### Task 1 — Fix Factorial Function (Easy)
The agent receives a factorial function with **syntax errors** (missing colon, wrong assignment operator in condition).  
**Reward:** fraction of 4 test cases passing.

### Task 2 — Fix Second Largest Element (Medium)
The agent receives a function with a **logic error** that returns the largest element instead of the second largest.  
**Reward:** fraction of 5 test cases passing.

### Task 3 — Fix Binary Search Algorithm (Hard)
The agent receives a binary search with **4 distinct algorithmic bugs**: off-by-one boundary, wrong while condition, and two incorrect pointer updates.  
**Reward:** fraction of 7 test cases passing.

---

## Action & Observation Spaces

### Observation
```json
{
  "task_id": "task_hard_binary_search",
  "task_name": "Fix Binary Search Algorithm",
  "difficulty": "hard",
  "task_description": "...",
  "buggy_code": "def binary_search(arr, target):\n    ...",
  "error_hint": "Hint: There are 4 bugs — check ...",
  "step_count": 1
}
```

### Action
```json
{
  "fixed_code": "def binary_search(arr, target):\n    ..."
}
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Start episode (optional: `{"task_id": "..."}`) |
| `POST` | `/step` | Submit fix `{"fixed_code": "..."}` |
| `GET` | `/state` | Current episode state |

---

## Local Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/pypatch
cd pypatch

# Install
pip install -r requirements.txt

# Run server
uvicorn main:app --host 0.0.0.0 --port 7860

# In another terminal — run inference
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
export ENV_URL=http://localhost:7860
python inference.py
```

## Docker

```bash
docker build -t pypatch .
docker run -p 7860:7860 pypatch
```

---

## Reward Design

Partial credit is intentional. If an agent fixes the syntax error in Task 3 but misses two algorithmic bugs, it might pass 4/7 test cases and receive reward 0.57. This dense feedback signal makes PyPatch suitable for RL training, not just one-shot evaluation.

---

## Required Environment Variables (inference)

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key |
| `ENV_URL` | PyPatch server URL |

---

*Built for OpenEnv Round 1 — April 2026*
