from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env

from rl_agent.options_policy import (
    OPTION_SPACE,
    OptionsRLPolicy,
    option_from_index,
    option_is_valid,
    option_state_key,
)

DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-level Q(s, option) for the options RL agent.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--epsilon-start", type=float, default=0.6)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--max-option-steps", type=int, default=80)
    parser.add_argument("--model-out", type=Path, default=Path(__file__).resolve().parent / "models" / "options_q.pkl")
    return parser.parse_args()


def epsilon_for_episode(index: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    progress = index / float(total - 1)
    return end + (start - end) * (1.0 - progress) ** 2


def choose_option(q_values: np.ndarray, valid: list[int], epsilon: float, rng: np.random.Generator) -> int:
    if not valid:
        return int(rng.integers(len(OPTION_SPACE)))
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    return max(valid, key=lambda index: float(q_values[index]))


def valid_options(policy: OptionsRLPolicy, state, info) -> list[int]:
    return [
        index
        for index in range(len(OPTION_SPACE))
        if option_is_valid(option_from_index(index), state, info, policy)
    ]


def run_option(env, policy: OptionsRLPolicy, obs, info, option_index: int, max_steps: int):
    option = option_from_index(option_index)
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0
    start_room = info.get("env", {}).get("room_id")
    start_events = dict(info.get("events", {}).get("counts", {}))

    while steps < max_steps and not (terminated or truncated):
        if policy.pending_interact:
            action = 5
        elif policy.committed_action is not None:
            action = policy.act(obs, info)
        else:
            state = policy._extract_state(obs, info)
            policy._update_memory(state, info)
            action, finished = policy._execute_option(option, state, info)
            if finished and steps > 0:
                break
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        room_changed = info.get("env", {}).get("room_id") != start_room
        event_counts = info.get("events", {}).get("counts", {})
        made_event = any(event_counts.get(name, 0) > start_events.get(name, 0) for name in event_counts)
        if room_changed or made_event:
            break

    return obs, info, total_reward, terminated, truncated, steps


def main() -> None:
    args = parse_args()
    q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(OPTION_SPACE), dtype=np.float32))
    rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, Any]] = []

    for task_index, task_id in enumerate(args.tasks):
        for episode in range(args.episodes):
            env = make_env(task_id=task_id, control_mode="pixel", observation_mode="grid")
            policy = OptionsRLPolicy()
            seed = args.seed + task_index * 100_000 + episode
            policy.reset(seed=seed, task_id=task_id)
            obs, info = env.reset(seed=seed)
            total_reward = 0.0
            terminated = False
            truncated = False
            option_steps = 0
            epsilon = epsilon_for_episode(episode, args.episodes, args.epsilon_start, args.epsilon_end)

            try:
                while not (terminated or truncated):
                    state = policy._extract_state(obs, info)
                    policy._update_memory(state, info)
                    key = option_state_key(state, info, policy)
                    valid = valid_options(policy, state, info)
                    action_index = choose_option(q_table[key], valid, epsilon, rng)
                    next_obs, next_info, reward, terminated, truncated, steps = run_option(
                        env,
                        policy,
                        obs,
                        info,
                        action_index,
                        args.max_option_steps,
                    )
                    next_state = policy._extract_state(next_obs, next_info)
                    policy._update_memory(next_state, next_info)
                    next_key = option_state_key(next_state, next_info, policy)
                    bootstrap = 0.0 if terminated else float(np.max(q_table[next_key]))
                    target = reward + args.gamma * bootstrap
                    q_table[key][action_index] += args.alpha * (target - q_table[key][action_index])
                    obs, info = next_obs, next_info
                    total_reward += reward
                    option_steps += 1
                    if steps == 0:
                        break
            finally:
                env.close()

            success = bool(info.get("game", {}).get("world_completed", False) or info.get("terminal_reason") == "world_completed")
            metrics.append({"task_id": task_id, "episode": episode, "success": success, "reward": total_reward})
            if (episode + 1) % max(1, args.episodes // 10) == 0:
                window = metrics[-max(1, args.episodes // 10):]
                task_window = [item for item in window if item["task_id"] == task_id]
                if task_window:
                    avg_reward = sum(item["reward"] for item in task_window) / len(task_window)
                    success_rate = sum(item["success"] for item in task_window) / len(task_window)
                    print(
                        f"{task_id} episode={episode + 1}/{args.episodes} "
                        f"success_rate={success_rate:.2f} avg_reward={avg_reward:.2f} states={len(q_table)}"
                    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "option_q_learning",
        "option_space": list(OPTION_SPACE),
        "tasks": list(args.tasks),
        "q_table": {key: values.astype(np.float32) for key, values in q_table.items()},
    }
    with args.model_out.open("wb") as file:
        pickle.dump(payload, file)
    print(f"saved model: {args.model_out}")


if __name__ == "__main__":
    main()


