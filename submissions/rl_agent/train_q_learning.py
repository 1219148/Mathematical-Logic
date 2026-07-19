from __future__ import annotations

import argparse
import json
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

try:
    from .features import state_key_from_grid_obs
except ImportError:
    from features import state_key_from_grid_obs


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for NesyLink.")
    parser.add_argument("--tasks", nargs="+", default=["mathematical_logic/task_1"], choices=DEFAULT_TASKS)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model-out", type=Path, default=Path(__file__).resolve().parent / "models" / "q_table.pkl")
    parser.add_argument("--metrics-out", type=Path, default=Path(__file__).resolve().parent / "models" / "training_metrics.json")
    return parser.parse_args()


def epsilon_for_episode(index: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    progress = index / float(total - 1)
    return end + (start - end) * (1.0 - progress) ** 2


def choose_action(q_table: dict[Any, np.ndarray], key: Any, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(7))
    return int(np.argmax(q_table[key]))


def train_task(
    *,
    task_id: str,
    q_table: dict[Any, np.ndarray],
    episodes: int,
    seed: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    env = make_env(
        task_id=task_id,
        control_mode="grid",
        observation_mode="grid",
        max_steps=max_steps,
    )
    metrics: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            epsilon = epsilon_for_episode(episode, episodes, epsilon_start, epsilon_end)
            obs, info = env.reset(seed=episode_seed)
            key = state_key_from_grid_obs(obs, info, task_id)
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            while not (terminated or truncated):
                action = choose_action(q_table, key, epsilon, rng)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                next_key = state_key_from_grid_obs(next_obs, next_info, task_id)

                bootstrap = 0.0 if terminated else float(np.max(q_table[next_key]))
                target = float(reward) + gamma * bootstrap
                q_table[key][action] += alpha * (target - q_table[key][action])

                key = next_key
                total_reward += float(reward)
                steps += 1

            metrics.append(
                {
                    "task_id": task_id,
                    "episode": episode,
                    "seed": episode_seed,
                    "epsilon": epsilon,
                    "steps": steps,
                    "total_reward": total_reward,
                    "success": bool(next_info.get("game", {}).get("world_completed", False)),
                    "terminal_reason": next_info.get("terminal_reason"),
                }
            )
            if (episode + 1) % max(1, episodes // 10) == 0:
                window = metrics[-max(1, episodes // 10) :]
                avg_reward = sum(item["total_reward"] for item in window) / len(window)
                success_rate = sum(item["success"] for item in window) / len(window)
                print(
                    f"{task_id} episode={episode + 1}/{episodes} "
                    f"avg_reward={avg_reward:.3f} success_rate={success_rate:.2f} "
                    f"states={len(q_table)}"
                )
    finally:
        env.close()

    return metrics


def save_model(path: Path, q_table: dict[Any, np.ndarray], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "tabular_q_learning",
        "execution_mode": "grid_to_pixel",
        "tasks": list(args.tasks),
        "q_table": {key: values.astype(np.float32) for key, values in q_table.items()},
        "args": vars(args) | {"model_out": str(args.model_out), "metrics_out": str(args.metrics_out)},
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    q_table: dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(7, dtype=np.float32))
    all_metrics: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(args.tasks):
        task_metrics = train_task(
            task_id=task_id,
            q_table=q_table,
            episodes=args.episodes,
            seed=args.seed + task_index * 10_000,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            max_steps=args.max_steps,
        )
        all_metrics.extend(task_metrics)

    save_model(args.model_out, q_table, args)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved model: {args.model_out}")
    print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()



