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
    task_cards = "".join(
        f"""<div class="task-card">
          <div class="task-top">
            <div>
              <h3>{t['name']}</h3>
              <p>{t['description'].splitlines()[0]}</p>
            </div>
            <span class="badge badge-{t['difficulty']}">{t['difficulty']}</span>
          </div>
          <div class="task-meta">
            <span><strong>ID</strong> <code>{t['id']}</code></span>
            <span><strong>Baseline</strong> {t['score']:.2f}</span>
            <span><strong>Grader</strong> <code>{t['grader']}</code></span>
          </div>
        </div>"""
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
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:radial-gradient(circle at top,#1a2238 0,#0f1117 45%);color:#e2e8f0;min-height:100vh;padding:2rem;line-height:1.55}}
  a{{color:#7dd3fc;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .shell{{max-width:1120px;margin:0 auto}}
  .hero{{text-align:center;padding:3.5rem 1rem 2rem}}
  .hero h1{{font-size:3.4rem;font-weight:800;background:linear-gradient(135deg,#fb923c,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .hero p{{color:#94a3b8;font-size:1.1rem;margin-top:.75rem;max-width:780px;margin-left:auto;margin-right:auto}}
  .hero-actions{{display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;margin-top:1.4rem}}
  .btn{{display:inline-flex;align-items:center;justify-content:center;padding:.8rem 1.1rem;border-radius:10px;font-weight:600;border:1px solid #334155;background:#111827;color:#f8fafc}}
  .btn-primary{{background:linear-gradient(135deg,#f97316,#ef4444);border:none}}
  .badge{{padding:.2rem .6rem;border-radius:999px;font-size:.75rem;font-weight:600;text-transform:uppercase}}
  .badge-easy{{background:#166534;color:#86efac}}
  .badge-medium{{background:#92400e;color:#fde68a}}
  .badge-hard{{background:#7f1d1d;color:#fca5a5}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;max-width:1120px;margin:2rem auto}}
  .card,.panel,.task-card{{background:rgba(30,35,48,.88);border:1px solid #2d3748;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,.22)}}
  .card{{padding:1.5rem}}
  .card h3{{font-size:.8rem;text-transform:uppercase;color:#64748b;letter-spacing:.1em;margin-bottom:.5rem}}
  .card p{{font-size:1.8rem;font-weight:700;color:#f1f5f9}}
  .card small{{font-size:.85rem;color:#64748b}}
  .section{{max-width:1120px;margin:2.3rem auto}}
  .section h2{{color:#f8fafc;font-size:1.45rem;margin-bottom:.35rem}}
  .section p.lead{{color:#94a3b8;margin-bottom:1rem}}
  .split{{display:grid;grid-template-columns:1.2fr .8fr;gap:1rem}}
  .panel{{padding:1.25rem}}
  .task-list{{display:grid;gap:1rem}}
  .task-card{{padding:1.2rem}}
  .task-top{{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start}}
  .task-card h3{{font-size:1.05rem;margin-bottom:.35rem}}
  .task-card p{{color:#94a3b8;font-size:.95rem}}
  .task-meta{{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem;color:#cbd5e1;font-size:.88rem}}
  code{{background:#0f1117;padding:.15rem .4rem;border-radius:4px;font-size:.82rem;color:#f97316}}
  pre{{background:#0b1220;border:1px solid #22304a;border-radius:12px;padding:1rem;overflow:auto;color:#cbd5e1;font-size:.88rem}}
  .endpoints{{max-width:1120px;margin:2rem auto}}
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
  .list{{display:grid;gap:.7rem;color:#cbd5e1}}
  .list div{{padding:.75rem .9rem;background:#111827;border:1px solid #243244;border-radius:10px}}
  footer{{text-align:center;color:#475569;font-size:.8rem;margin-top:3rem}}
  @media (max-width: 840px){{.split{{grid-template-columns:1fr}} .hero h1{{font-size:2.6rem}}}}
</style>
</head>
<body>
<div class="shell">
<div class="hero">
  <h1>🐛 PyPatch</h1>
  <p>PyPatch is a judge-friendly RL environment where agents repair buggy Python programs across escalating difficulty levels and receive dense, test-driven reward signals.</p>
  <br/>
  <span class="status-pill"><span class="dot"></span>Live · v1.0.0</span>
  <div class="hero-actions">
    <a class="btn btn-primary" href="/docs">Open API Docs</a>
    <a class="btn" href="/tasks">View Task Metadata</a>
    <a class="btn" href="https://github.com/Anurag-M1/PyPatch">GitHub Repo</a>
  </div>
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
  <div class="card">
    <h3>Baseline</h3>
    <p>0.67</p>
    <small>deterministic average score</small>
  </div>
</div>

<div class="section">
  <h2>Why This Stands Out</h2>
  <p class="lead">Designed for fast evaluation, dense feedback, and clear progression from syntax repair to algorithmic reasoning.</p>
  <div class="split">
    <div class="panel">
      <div class="list">
        <div><strong>Dense reward shaping</strong><br/>Agents receive partial credit instead of only pass/fail outcomes.</div>
        <div><strong>Progressive difficulty</strong><br/>Easy, medium, and hard tasks cover syntax, logic, and algorithmic bug fixing.</div>
        <div><strong>Deterministic grading</strong><br/>Each task uses explicit hidden test cases and reproducible scoring.</div>
        <div><strong>Submission-safe baseline</strong><br/>Baseline inference uses the evaluator proxy and emits structured logs.</div>
      </div>
    </div>
    <div class="panel">
<pre>GET  /        # machine-readable health
GET  /ui     # judge-facing landing page
GET  /tasks  # task metadata + grader + score
POST /reset  # start a task episode
POST /step   # submit fixed code
GET  /state  # inspect current episode</pre>
    </div>
  </div>
</div>

<div class="section">
  <h2>Tasks</h2>
  <p class="lead">Three curated debugging challenges with deterministic graders and non-trivial hidden tests.</p>
  <div class="task-list">{task_cards}</div>
</div>

<div class="section">
  <h2>Judge Demo Flow</h2>
  <p class="lead">A reviewer can understand the project in under two minutes.</p>
  <div class="split">
    <div class="panel">
<pre>1. Open /ui
2. Inspect /tasks for difficulty + grader metadata
3. POST /reset with a task_id
4. POST /step with corrected Python code
5. Observe reward and test breakdown
6. Run inference.py for baseline logs</pre>
    </div>
    <div class="panel">
<pre>curl -X POST http://localhost:7860/reset \\
  -H "Content-Type: application/json" \\
  -d '{{"task_id":"task_medium_second_largest"}}'

curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{{"fixed_code":"def second_largest(lst):\\n    unique = sorted(set(lst))\\n    return unique[-2]"}}'</pre>
    </div>
  </div>
</div>

<div class="endpoints">
  <h2>API Endpoints</h2>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/ui</span><span class="ep-desc">Judge-facing landing page</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/health</span><span class="ep-desc">Health check</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/tasks</span><span class="ep-desc">List all tasks</span></div>
  <div class="ep"><span class="method post">POST</span><span class="ep-path">/reset</span><span class="ep-desc">Start episode · body: {{"task_id": "..."}}</span></div>
  <div class="ep"><span class="method post">POST</span><span class="ep-path">/step</span><span class="ep-desc">Submit fix · body: {{"fixed_code": "..."}}</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/state</span><span class="ep-desc">Current episode state</span></div>
  <div class="ep"><span class="method get">GET</span><span class="ep-path">/docs</span><span class="ep-desc">Swagger UI</span></div>
</div>

<footer>Built by Anurag Singh</footer>
</div>
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
                grader=t["grader"],
                score=t["score"],
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
