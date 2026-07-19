from __future__ import annotations

import argparse
import importlib
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
from submissions.perception import PerceptionEngine

try:
    from .features import state_key_from_symbolic_state
except ImportError:
    from features import state_key_from_symbolic_state


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a tabular Q policy from expert pixel rollouts.")
    parser.add_argument("--expert", default="submissions.student_agent", help="Expert module exposing make_policy().")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds per task.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-q", type=float, default=1000.0)
    parser.add_argument("--model-out", type=Path, default=Path(__file__).resolve().parent / "models" / "q_table.pkl")
    return parser.parse_args()


def load_expert(spec: str) -> Any:
    module = importlib.import_module(spec)
    if hasattr(module, "make_policy"):
        return module.make_policy()
    if hasattr(module, "Policy"):
        return module.Policy()
    if hasattr(module, "policy"):
        return module.policy
    raise AttributeError(f"expert module {spec!r} must expose make_policy, Policy, or policy")


def reset_policy(policy: Any, *, seed: int, task_id: str) -> None:
    reset = getattr(policy, "reset", None)
    if reset is None:
        return
    try:
        reset(seed=seed, task_id=task_id)
    except TypeError:
        try:
            reset(seed=seed)
        except TypeError:
            reset()


def call_policy(policy: Any, obs: np.ndarray, info: dict[str, Any]) -> int:
    actor = policy.act if hasattr(policy, "act") else policy
    try:
        action = actor(obs, info)
    except TypeError:
        action = actor(obs)
    if isinstance(action, dict):
        action = action.get("action")
    if isinstance(action, (tuple, list)) and action:
        action = action[0]
    return int(np.asarray(action).item())


def main() -> None:
    args = parse_args()
    q_table: dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(7, dtype=np.float32))
    perception = PerceptionEngine(device="cpu")
    total_steps = 0
    successes = 0

    for task_id in args.tasks:
        for seed_offset in range(args.seeds):
            seed = args.seed + seed_offset
            expert = load_expert(args.expert)
            reset_policy(expert, seed=seed, task_id=task_id)
            env = make_env(task_id=task_id, observation_mode="pixels")
            obs, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            steps = 0
            try:
                while not (terminated or truncated):
                    state = perception.extract(obs)
                    key = state_key_from_symbolic_state(state, info, task_id)
                    action = call_policy(expert, obs, info)
                    q_table[key][action] = max(q_table[key][action], args.expert_q)
                    obs, reward, terminated, truncated, info = env.step(action)
                    del reward
                    steps += 1
            finally:
                env.close()
            success = bool(info.get("game", {}).get("world_completed", False) or info.get("terminal_reason") == "world_completed")
            successes += int(success)
            total_steps += steps
            print(f"{task_id} seed={seed} success={success} steps={steps} states={len(q_table)}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "expert_initialized_tabular_q",
        "execution_mode": "grid_to_pixel",
        "tasks": list(args.tasks),
        "q_table": {key: values.astype(np.float32) for key, values in q_table.items()},
        "expert": args.expert,
    }
    with args.model_out.open("wb") as file:
        pickle.dump(payload, file)
    print(f"saved model: {args.model_out}")
    print(f"expert successes: {successes}/{len(args.tasks) * args.seeds}, total_steps={total_steps}")


if __name__ == "__main__":
    main()

