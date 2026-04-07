from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from focus_resource_env import (
    DEEP_WORK,
    EMPTY,
    MEETING,
    OP_IDLE,
    OP_MUTE_COMMS,
    OP_RESCHEDULE_MEETING,
    OP_SCHEDULE_WORK,
    FocusResourceEnv,
)


def choose_action(env: FocusResourceEnv) -> Tuple[int, int]:
    current = env.current_slot

    if env.current_slot == 0 and not env.mute_comms and env.distraction_risk > 0.0:
        return current, OP_MUTE_COMMS

    if env.recovery_remaining > 0:
        return current, OP_IDLE

    if current < env.timeline_length and env.timeline[current] == EMPTY:
        return current, OP_SCHEDULE_WORK

    empty_slots = np.where(env.timeline[current:] == EMPTY)[0]
    if empty_slots.size > 0:
        target = current + int(empty_slots[0])
        return target, OP_SCHEDULE_WORK

    fragmented_meetings = future_meeting_starts(env, current)
    if fragmented_meetings:
        _, largest_empty_len = largest_empty_block(env.timeline, current)
        if largest_empty_len < 8:
            return fragmented_meetings[0], OP_RESCHEDULE_MEETING

    return current, OP_IDLE


def largest_empty_block(timeline: np.ndarray, start_index: int) -> Tuple[int, int]:
    best_start = start_index
    best_len = 0
    idx = start_index
    while idx < len(timeline):
        if timeline[idx] != EMPTY:
            idx += 1
            continue
        run_start = idx
        while idx < len(timeline) and timeline[idx] == EMPTY:
            idx += 1
        run_len = idx - run_start
        if run_len > best_len:
            best_start = run_start
            best_len = run_len
    return best_start, best_len


def future_meeting_starts(env: FocusResourceEnv, current: int) -> list[int]:
    starts = []
    seen = set()
    for slot in range(current, env.timeline_length):
        if env.timeline[slot] != MEETING:
            continue
        meta = env.meeting_meta.get(slot)
        if meta is None:
            continue
        start = meta["start"]
        if start in seen or start < current:
            continue
        seen.add(start)
        starts.append(start)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FocusResourceEnv simulation.")
    parser.add_argument("--start-hour", default="09:00")
    parser.add_argument("--end-hour", default="17:00")
    parser.add_argument("--distraction-risk", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = FocusResourceEnv(
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        distraction_risk=args.distraction_risk,
        seed=args.seed,
    )

    observation = env.reset()
    total_reward = 0.0
    print("Initial observation:")
    print(observation)
    print(env.render_text())

    done = False
    while not done:
        action = choose_action(env)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(f"action={action} reward={reward:.2f}")
        print(info)
        print(env.render_text())

    print("\nFinal Flow Efficiency score")
    print(f"episode_reward={total_reward:.2f}")
    print(f"flow_score={env.flow_score:.2f}")
    print(f"social_debt={env.social_debt:.2f}")
    print(f"calendar_churn={env.calendar_churn}")
    print(f"interruptions={env.interruptions}")
    print(f"invalid_actions={env.invalid_actions}")


if __name__ == "__main__":
    main()
