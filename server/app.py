"""FastAPI application for the Engineer Manager OpenEnv server."""

from __future__ import annotations

from textwrap import dedent

import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from openenv.core.env_server.http_server import create_fastapi_app
from pydantic import BaseModel

from graders import grade_task_0, grade_task_1, grade_task_2
from tasks import TASKS

try:
    from ..models import EngineerManagerAction, EngineerManagerObservation
    from .engineer_manager_environment import EngineerManagerEnvironment
except ImportError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from models import EngineerManagerAction, EngineerManagerObservation
    from server.engineer_manager_environment import EngineerManagerEnvironment


app = create_fastapi_app(
    EngineerManagerEnvironment,
    EngineerManagerAction,
    EngineerManagerObservation,
    max_concurrent_envs=2,
)


class GraderRequest(BaseModel):
    task_id: str
    state: dict
    reward: float

WEB_CSS = dedent(
    """\
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(10, 21, 38, 0.88);
      --panel-2: rgba(14, 28, 48, 0.96);
      --line: rgba(124, 231, 255, 0.18);
      --text: #eaf4ff;
      --muted: #99abc7;
      --accent: #7ce7ff;
      --accent-2: #86f0ca;
      --meeting: #f5c76a;
      --empty: rgba(255, 255, 255, 0.06);
      --danger: #ff8d8d;
      --shell-max: 1180px;
      --sidebar-width: 300px;
      --content-min: 620px;
    }
    * { box-sizing: border-box; }
    html {
      min-height: 100%;
      background: #07111f;
    }
    body {
      margin: 0;
      min-height: 100%;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(124, 231, 255, 0.12), transparent 30rem),
        linear-gradient(180deg, #09111d 0%, #07111f 100%);
    }
    .shell {
      width: min(100%, var(--shell-max));
      margin: 0 auto;
      padding: 32px 20px 40px;
    }
    .hero {
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(9, 20, 38, 0.95), rgba(12, 31, 54, 0.92));
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 5vw, 3.8rem);
      line-height: 0.95;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      max-width: 62ch;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(260px, var(--sidebar-width)) minmax(var(--content-min), 1fr);
      align-items: start;
      gap: 18px;
      margin-top: 18px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.22);
      backdrop-filter: blur(10px);
    }
    .panel-inner { padding: 18px; }
    .section-title {
      margin: 0 0 14px;
      font-size: 0.92rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric {
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-2);
    }
    .metric small {
      display: block;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.7rem;
    }
    .metric strong {
      font-size: 1.35rem;
    }
    label {
      display: block;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    input, select, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    button {
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease;
      background: linear-gradient(135deg, rgba(124, 231, 255, 0.22), rgba(134, 240, 202, 0.16));
    }
    button:hover { transform: translateY(-1px); }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .actions button[data-op="0"] { grid-column: span 2; }
    .timeline {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(58px, 1fr));
      gap: 8px;
    }
    .slot {
      border-radius: 16px;
      min-height: 84px;
      padding: 10px 8px;
      border: 1px solid transparent;
      background: var(--empty);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .slot.current { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(124, 231, 255, 0.25) inset; }
    .slot.work { background: linear-gradient(180deg, rgba(124, 240, 202, 0.22), rgba(124, 240, 202, 0.10)); }
    .slot.meeting { background: linear-gradient(180deg, rgba(245, 199, 106, 0.28), rgba(245, 199, 106, 0.10)); }
    .slot small { color: var(--muted); }
    .slot strong { font-size: 0.82rem; }
    .task-list, pre {
      margin: 0;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.03);
    }
    .task-list li { margin: 0 0 8px; color: var(--muted); }
    .task-list li:last-child { margin-bottom: 0; }
    pre {
      overflow: auto;
      max-height: 260px;
      color: #d7e6ff;
    }
    .details-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .status {
      margin-top: 12px;
      min-height: 24px;
      color: var(--muted);
    }
    .status.error { color: var(--danger); }
    @media (max-width: 1040px) {
      :root {
        --content-min: 0px;
      }
      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 760px) {
      .grid,
      .metrics,
      .details-grid {
        grid-template-columns: 1fr;
      }
      .actions { grid-template-columns: 1fr; }
      .actions button[data-op="0"] { grid-column: span 1; }
    }
    """
)

WEB_JS = dedent(
    """\
    const statusEl = document.getElementById("status");
    const targetSlotEl = document.getElementById("targetSlot");
    const timelineEl = document.getElementById("timeline");
    const tasksEl = document.getElementById("tasks");
    const payloadEl = document.getElementById("payload");
    const metrics = {
      time: document.getElementById("m-time"),
      slot: document.getElementById("m-slot"),
      flow: document.getElementById("m-flow"),
      debt: document.getElementById("m-debt"),
    };

    function setStatus(message, isError = false) {
      statusEl.textContent = message;
      statusEl.className = isError ? "status error" : "status";
    }

    function kindLabel(value) {
      if (value === 1) return "Deep Work";
      if (value === 2) return "Meeting";
      return "Open";
    }

    function renderObservation(response) {
      const obs = response.observation || {};
      metrics.time.textContent = obs.current_time ?? "-";
      metrics.slot.textContent = String(obs.current_slot ?? "-");
      metrics.flow.textContent = Number(obs.flow_score ?? 0).toFixed(2);
      metrics.debt.textContent = Number(obs.social_debt ?? 0).toFixed(2);
      targetSlotEl.max = Math.max(0, (obs.timeline || []).length - 1);

      timelineEl.innerHTML = "";
      (obs.timeline || []).forEach((value, index) => {
        const slot = document.createElement("div");
        slot.className = "slot " + (value === 1 ? "work" : value === 2 ? "meeting" : "");
        if (index === obs.current_slot) slot.classList.add("current");
        slot.innerHTML = "<small>Slot " + index + "</small><strong>" + kindLabel(value) + "</strong>";
        timelineEl.appendChild(slot);
      });

      tasksEl.innerHTML = "";
      const tasks = obs.task_buffer || [];
      if (!tasks.length) {
        const empty = document.createElement("li");
        empty.textContent = "No queued tasks.";
        tasksEl.appendChild(empty);
      } else {
        tasks.forEach((task, index) => {
          const item = document.createElement("li");
          item.textContent = "Task " + (index + 1) + ": " + task.duration + " slots, complexity " + task.hidden_complexity;
          tasksEl.appendChild(item);
        });
      }

      payloadEl.textContent = JSON.stringify(response, null, 2);
    }

    async function callJson(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      return data;
    }

    async function resetEnv() {
      setStatus("Resetting environment...");
      try {
        const data = await callJson("/reset", {});
        renderObservation(data);
        setStatus("Environment reset.");
      } catch (error) {
        setStatus("Reset failed: " + error.message, true);
      }
    }

    async function stepEnv(operation) {
      setStatus("Applying action...");
      try {
        const payload = {
          action: {
            target_slot: Number(targetSlotEl.value || 0),
            operation: Number(operation),
          },
        };
        const data = await callJson("/step", payload);
        renderObservation(data);
        setStatus("Action applied.");
      } catch (error) {
        setStatus("Action failed: " + error.message, true);
      }
    }

    document.getElementById("resetBtn").addEventListener("click", resetEnv);
    document.querySelectorAll("[data-op]").forEach((button) => {
      button.addEventListener("click", () => stepEnv(button.dataset.op));
    });

    resetEnv();
    """
)

WEB_PAGE = dedent(
    """\
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Engineer Manager</title>
      <link rel="icon" href="/favicon.ico" sizes="any">
      <link rel="stylesheet" href="/assets/app.css">
    </head>
    <body>
      <div class="shell">
        <section class="hero">
          <div class="eyebrow">Engineer Manager Simulator</div>
          <h1>Run the day before the day runs you.</h1>
          <p class="sub">This is a direct native UI for the Space. It talks to the environment API without the Gradio bridge, so it avoids the Hugging Face wrapper errors that were breaking the previous page.</p>
        </section>

        <div class="grid">
          <aside class="panel">
            <div class="panel-inner">
              <h2 class="section-title">Controls</h2>
              <label>Target slot
                <input id="targetSlot" type="number" min="0" value="0">
              </label>
              <button id="resetBtn">Reset Environment</button>
              <div class="actions">
                <button data-op="0">Advance Time</button>
                <button data-op="1">Schedule Work</button>
                <button data-op="2">Move Meeting</button>
                <button data-op="3">Mute Comms</button>
              </div>
              <div id="status" class="status">Ready.</div>
            </div>
          </aside>

          <main class="panel">
            <div class="panel-inner">
              <div class="metrics">
                <div class="metric"><small>Current Time</small><strong id="m-time">-</strong></div>
                <div class="metric"><small>Current Slot</small><strong id="m-slot">-</strong></div>
                <div class="metric"><small>Flow Score</small><strong id="m-flow">-</strong></div>
                <div class="metric"><small>Social Debt</small><strong id="m-debt">-</strong></div>
              </div>

              <h2 class="section-title">Timeline</h2>
              <div id="timeline" class="timeline"></div>

              <div class="details-grid">
                <section>
                  <h2 class="section-title">Task Buffer</h2>
                  <ul id="tasks" class="task-list"></ul>
                </section>
                <section>
                  <h2 class="section-title">Last Response</h2>
                  <pre id="payload">{}</pre>
                </section>
              </div>
            </div>
          </main>
        </div>
      </div>
      <script src="/assets/app.js" defer></script>
    </body>
    </html>
    """
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web/")


@app.get("/web", include_in_schema=False)
def web_root() -> RedirectResponse:
    return RedirectResponse(url="/web/")


@app.get("/web/", include_in_schema=False)
def web_ui() -> HTMLResponse:
    return HTMLResponse(WEB_PAGE)


@app.get("/assets/app.css", include_in_schema=False)
def web_css() -> PlainTextResponse:
    return PlainTextResponse(WEB_CSS, media_type="text/css")


@app.get("/assets/app.js", include_in_schema=False)
def web_js() -> PlainTextResponse:
    return PlainTextResponse(WEB_JS, media_type="application/javascript")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/manifest.json", include_in_schema=False)
def manifest() -> JSONResponse:
    return JSONResponse(
        {
            "name": "Engineer Manager",
            "short_name": "Engineer Manager",
            "start_url": "/web/",
            "display": "standalone",
            "background_color": "#0b1224",
            "theme_color": "#0b1224",
            "icons": [],
        }
    )


@app.get("/tasks", include_in_schema=False)
def tasks() -> list[dict]:
    return TASKS


@app.post("/grader", include_in_schema=False)
def grader(request: GraderRequest) -> JSONResponse:
    task_index_map = {
        "quiet-morning": 0,
        "engineer_manager_task_0": 0,
        "meeting-surgery": 1,
        "engineer_manager_task_1": 1,
        "delivery-triage": 2,
        "engineer_manager_task_2": 2,
    }
    grader_fn_map = {
        "quiet-morning": grade_task_0,
        "meeting-surgery": grade_task_1,
        "delivery-triage": grade_task_2,
        "engineer_manager_task_0": grade_task_0,
        "engineer_manager_task_1": grade_task_1,
        "engineer_manager_task_2": grade_task_2,
    }
    grader_fn = grader_fn_map.get(request.task_id)
    if grader_fn is None:
        return JSONResponse(
            {"error": f"Unknown task_id: {request.task_id}", "score": 0.0, "passed": False},
            status_code=400,
        )
    state = dict(request.state)
    if "task_id" not in state or state["task_id"] is None:
        state["task_id"] = task_index_map[request.task_id]
    score = float(grader_fn(state, request.reward))
    return JSONResponse({"task_id": request.task_id, "score": score, "passed": score >= 0.0, "reward": score})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv HTTP server."""
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI entrypoint expected by the OpenEnv validator."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
