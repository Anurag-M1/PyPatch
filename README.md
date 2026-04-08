# PyPatch

PyPatch is a FastAPI-based reinforcement learning environment where agents repair buggy Python code and receive partial reward based on hidden test-case performance.

Built by Anurag Singh.

## Overview

- 3 tasks across `easy`, `medium`, and `hard`
- Up to 5 attempts per episode
- Reward range from `0.0` to `1.0`
- REST API for reset, step, task listing, and state inspection
- Local validator included for pre-submission checks

## Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| Fix Factorial Function | Easy | Repair syntax issues in a recursive factorial function |
| Fix Second Largest Element | Medium | Correct logic so the second-largest element is returned |
| Fix Binary Search Algorithm | Hard | Repair multiple algorithmic bugs in binary search |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Landing page |
| `GET` | `/health` | Service health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit a code fix |
| `GET` | `/state` | Inspect current episode state |

### Example observation

```json
{
  "task_id": "task_hard_binary_search",
  "task_name": "Fix Binary Search Algorithm",
  "difficulty": "hard",
  "task_description": "...",
  "buggy_code": "def binary_search(arr, target):\n    ...",
  "error_hint": "Hint: There are multiple algorithmic bugs to fix.",
  "step_count": 1
}
```

### Example action

```json
{
  "fixed_code": "def binary_search(arr, target):\n    ..."
}
```

## Run Locally

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open:

- `http://127.0.0.1:7860`
- `http://127.0.0.1:7860/docs`

## Validate

With the server already running:

```bash
python3 validate.py
```

Or let the validator start the server:

```bash
python3 validate.py --start-server
```

## Inference Script

`inference.py` is included for model-driven interaction with the environment.

Typical environment variables:

| Variable | Description |
| --- | --- |
| `API_BASE_URL` | Model API base URL |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key or token |
| `ENV_URL` | PyPatch server URL, usually `http://localhost:7860` |

## Docker

```bash
docker build -t pypatch .
docker run -p 7860:7860 pypatch
```

## Project Files

- `main.py` - FastAPI app and endpoints
- `tasks.py` - task definitions and grading logic
- `models.py` - request and response schemas
- `validate.py` - local validation suite
- `inference.py` - sample agent runner
- `openenv.yaml` - environment manifest

## Notes

- The browser should use `127.0.0.1` or `localhost`, not `0.0.0.0`
- Reward is proportional to hidden tests passed, so partial fixes earn partial credit
- The app supports hot reload during development with `--reload`
