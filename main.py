import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from models import (
    PyPatchAction,
    PyPatchObservation,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
    TaskInfo,
)
from tasks import TASKS, TASK_MAP, grade_task

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PyPatch",
    description=(
        "An OpenEnv-compliant RL environment where AI agents fix buggy Python code "
        "across three difficulty levels: easy (syntax), medium (logic), hard (algorithmic)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_STEPS_PER_TASK = 5

# ─── SESSION STATE ────────────────────────────────────────────────────────────
# Single-session in-memory state (compatible with single-worker Docker deployment)

_state = {
    "session_id": None,
    "current_task": None,
    "step_count": 0,
    "done": False,
    "best_reward": 0.0,
    "last_reward": 0.0,
}


def _make_observation(task: dict, step_count: int) -> PyPatchObservation:
    return PyPatchObservation(
        task_id=task["id"],
        task_name=task["name"],
        difficulty=task["difficulty"],
        task_description=task["description"],
        buggy_code=task["buggy_code"],
        error_hint=task["error_hint"],
        step_count=step_count,
    )


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

def _landing_page_html() -> str:
    task_rows = "".join(
        f"""<tr>
          <td><code>{t['id']}</code></td>
          <td>{t['name']}</td>
          <td><span class="badge badge-{t['difficulty']}">{t['difficulty']}</span></td>
          <td>{t['description'].splitlines()[0][:80]}…</td>
        </tr>"""
        for t in TASKS
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PyPatch 🐛</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:2rem}}
  .hero{{text-align:center;padding:3rem 1rem 2rem}}
  .hero h1{{font-size:3rem;font-weight:800;background:linear-gradient(135deg,#f97316,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .hero p{{color:#94a3b8;font-size:1.1rem;margin-top:.75rem}}
  .badge{{padding:.2rem .6rem;border-radius:999px;font-size:.75rem;font-weight:600;text-transform:uppercase}}
  .badge-easy{{background:#166534;color:#86efac}}
  .badge-medium{{background:#92400e;color:#fde68a}}
  .badge-hard{{background:#7f1d1d;color:#fca5a5}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;max-width:900px;margin:2rem auto}}
  .card{{background:#1e2330;border:1px solid #2d3748;border-radius:12px;padding:1.5rem}}
  .card h3{{font-size:.8rem;text-transform:uppercase;color:#64748b;letter-spacing:.1em;margin-bottom:.5rem}}
  .card p{{font-size:1.8rem;font-weight:700;color:#f1f5f9}}
  .card small{{font-size:.85rem;color:#64748b}}
  table{{width:100%;max-width:900px;margin:1.5rem auto;border-collapse:collapse;background:#1e2330;border-radius:12px;overflow:hidden}}
  th{{background:#2d3748;padding:.75rem 1rem;text-align:left;font-size:.8rem;text-transform:uppercase;color:#94a3b8;letter-spacing:.05em}}
  td{{padding:.75rem 1rem;border-top:1px solid #2d3748;font-size:.875rem;color:#cbd5e1}}
  td code{{background:#0f1117;padding:.15rem .4rem;border-radius:4px;font-size:.8rem;color:#f97316}}
  .endpoints{{max-width:900px;margin:2rem auto}}
  .endpoints h2{{color:#94a3b8;font-size:.9rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:1rem}}
  .ep{{display:flex;align-items:center;gap:.75rem;padding:.6rem 1rem;background:#1e2330;border-radius:8px;margin-bottom:.5rem;border:1px solid #2d3748}}
  .method{{font-size:.75rem;font-weight:700;padding:.2rem .5rem;border-radius:4px;min-width:3.5rem;text-align:center}}
  .get{{background:#164e63;color:#7dd3fc}}
  .post{{background:#14532d;color:#86efac}}
  .ep-path{{font-family:monospace;font-size:.9rem;color:#e2e8f0}}
  .ep-desc{{font-size:.8rem;color:#64748b;margin-left:auto}}
  .status-pill{{display:inline-flex;align-items:center;gap:.4rem;background:#14532d;border:1px solid #166534;color:#86efac;padding:.3rem .8rem;border-radius:999px;font-size:.8rem;font-weight:600}}
  .dot{{width:8px;height:8px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
  footer{{text-align:center;color:#475569;font-size:.8rem;margin-top:3rem}}
</style>
</head>
<body>
<div class="hero">
  <h1>🐛 PyPatch</h1>
  <p>OpenEnv RL environment — AI agents debug &amp; fix Python code</p>
  <br/>
  <span class="status-pill"><span class="dot"></span>Live · v1.0.0</span>
</div>

<div class="grid">
  <div class="card">
    <h3>Tasks</h3>
    <p>{len(TASKS)}</p>
    <small>easy → medium → hard</small>
  </div>
  <div class="card">
    <h3>Max Steps</h3>
    <p>5</p>
    <small>per episode</small>
  </div>
  <div class="card">
    <h3>Reward</h3>
    <p>0–1</p>
    <small>fraction of tests passed</small>
  </div>
</div>

<table>
  <thead><tr><th>Task ID</th><th>Name</th><th>Difficulty</th><th>Description</th></tr></thead>
  <tbody>{task_rows}</tbody>
</table>

<div class="endpoints">
  <h2>API Endpoints</h2>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/health</span><span class="ep-desc">Health check</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/tasks</span><span class="ep-desc">List all tasks</span></div>
  <div class="ep"><span class="method post">POST</span><span class="ep-path">/reset</span><span class="ep-desc">Start episode · body: {{"task_id": "..."}}</span></div>
  <div class="ep"><span class="method post">POST</span><span class="ep-path">/step</span><span class="ep-desc">Submit fix · body: {{"fixed_code": "..."}}</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/state</span><span class="ep-desc">Current episode state</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/docs</span><span class="ep-desc">Swagger UI</span></div>
</div>

<footer>Built by Anurag Singh</footer>
</body>
</html>"""
    return html


@app.get("/")
async def root():
    """Machine-readable health endpoint used by validators."""
    return {
        "status": "ok",
        "env": "PyPatch",
        "version": app.version,
        "tasks": len(TASKS),
    }


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Human-friendly landing page."""
    return HTMLResponse(content=_landing_page_html(), status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            TaskInfo(
                id=t["id"],
                name=t["name"],
                difficulty=t["difficulty"],
                description=t["description"],
            )
            for t in TASKS
        ]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment.
    Optionally pass `task_id` to select a specific task.
    Defaults to the easy task.
    """
    task_id = request.task_id if request and request.task_id else TASKS[0]["id"]

    task = TASK_MAP.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    _state["session_id"] = str(uuid.uuid4())
    _state["current_task"] = task
    _state["step_count"] = 0
    _state["done"] = False
    _state["best_reward"] = 0.0
    _state["last_reward"] = 0.0

    return ResetResponse(
        observation=_make_observation(task, 0),
        done=False,
        reward=0.0,
    )


@app.post("/step", response_model=StepResponse)
async def step(action: PyPatchAction):
    """
    Submit a fixed version of the buggy code.
    Returns reward (0.0–1.0) based on test cases passed.
    Episode ends when all tests pass OR max steps reached.
    """
    if _state["current_task"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    if _state["done"]:
        raise HTTPException(status_code=400, detail="Episode complete. Call /reset to start a new episode.")

    _state["step_count"] += 1
    task = _state["current_task"]

    reward, info = grade_task(task, action.fixed_code)
    done = reward >= 1.0 or _state["step_count"] >= MAX_STEPS_PER_TASK

    _state["best_reward"] = max(_state["best_reward"], reward)
    _state["last_reward"] = reward
    _state["done"] = done

    return StepResponse(
        observation=_make_observation(task, _state["step_count"]),
        reward=reward,
        done=done,
        info={
            "tests_passed": info["passed"],
            "tests_total": info["total"],
            "test_results": info["results"],
            "exec_error": info.get("error"),
            "steps_remaining": max(0, MAX_STEPS_PER_TASK - _state["step_count"]),
        },
    )


@app.get("/state", response_model=StateResponse)
async def get_state():
    """Return current environment state."""
    return StateResponse(
        task_id=_state["current_task"]["id"] if _state["current_task"] else None,
        step_count=_state["step_count"],
        done=_state["done"],
        total_reward=_state["best_reward"],
        session_id=_state["session_id"],
    )
