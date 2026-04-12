---
title: Engineer Manager Environment Server
emoji: "🗂️"
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - scheduling
  - reinforcement-learning
---

# Engineer Manager Environment

`Engineer Manager` is an OpenEnv-compatible scheduling simulator for balancing deep work, meetings, and communication load across a workday.

## What it exposes

- `POST /reset` to start a fresh episode
- `POST /step` to apply a scheduling action
- `GET /state` for the current session state
- `GET /schema` for the action and observation schemas
- `GET /web` for the built-in OpenEnv web UI

## Local usage

```bash
python -m server.app --port 8000
openenv validate .
openenv validate http://127.0.0.1:8000
```

## Action model

- `target_slot`: target half-hour slot
- `operation`: `0` idle, `1` schedule work, `2` reschedule meeting, `3` mute comms

## Observation highlights

- `timeline`: day plan encoded as empty/work/meeting slots
- `task_buffer`: pending tasks with estimated duration and hidden complexity
- `flow_score`, `social_debt`, `calendar_churn`: core scoring metrics
- `current_slot`, `current_time`, `recovery_state`, `mute_comms`: live execution state

## Built-in benchmark tasks

Set `TASK_NAME` to select a deterministic scenario before reset. Available tasks:

- `quiet-morning`: high-noise start where muting comms early and protecting focus is rewarded
- `meeting-surgery`: fragmented calendar where selective meeting moves should improve flow
- `delivery-triage`: constrained delivery day with hidden task complexity and tighter tradeoffs

Each task has a grader in [benchmark_tasks.py](/C:/Users/arshi/OneDrive/Desktop/idk/engineer-manager/benchmark_tasks.py:1). The environment also exposes task metadata and the current grader score in `observation.metadata.episode_metrics.grader_score`.
