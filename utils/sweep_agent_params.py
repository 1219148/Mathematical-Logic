"""Search mechanism-level planner parameters for the submitted agent.

The sweep only changes generic planner costs and controller thresholds through
environment variables. It does not encode task-specific routes or room names.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK5_ID = "mathematical_logic/task_5"


@dataclass(frozen=True)
class SweepCase:
    name: str
    params: dict[str, float | int]


CASES: tuple[SweepCase, ...] = (
    SweepCase("baseline", {}),
    SweepCase("low_monster_risk", {"NESYLINK_MONSTER_DANGER_COST": 1.0}),
    SweepCase("no_monster_risk", {"NESYLINK_MONSTER_DANGER_COST": 0.0}),
    SweepCase("high_monster_risk", {"NESYLINK_MONSTER_DANGER_COST": 4.0}),
    SweepCase("low_trap_risk", {"NESYLINK_TRAP_NEIGHBOR_COST": 2.0}),
    SweepCase("high_trap_risk", {"NESYLINK_TRAP_NEIGHBOR_COST": 8.0}),
    SweepCase("explore_unknown_more", {"NESYLINK_UNKNOWN_EXIT_BONUS": 60.0}),
    SweepCase("retry_key_blocked_more", {"NESYLINK_KEY_BLOCKED_RETRY_BONUS": 120.0}),
    SweepCase("shorter_shield_cooldown", {"NESYLINK_SHIELD_COOLDOWN_TICKS": 10}),
    SweepCase("farther_shield", {"NESYLINK_SHIELD_DISTANCE": 3}),
    SweepCase("balanced_fast", {
        "NESYLINK_MONSTER_DANGER_COST": 1.0,
        "NESYLINK_TRAP_NEIGHBOR_COST": 2.0,
        "NESYLINK_UNKNOWN_EXIT_BONUS": 60.0,
        "NESYLINK_KEY_BLOCKED_RETRY_BONUS": 100.0,
        "NESYLINK_SHIELD_COOLDOWN_TICKS": 12,
        "NESYLINK_STUCK_RECOVERY_TICKS": 20,
    }),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small parameter sweep for submissions/student_agent.py.")
    parser.add_argument("--policy", default="submissions/student_agent.py")
    parser.add_argument("--tasks", nargs="+", default=[TASK5_ID])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=len(CASES), help="Maximum number of sweep cases to run.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/agent_param_sweep"))
    return parser.parse_args()


def task_score(task_summary: dict[str, Any]) -> float:
    milestones = task_summary.get("milestone_rates", {})
    events = task_summary.get("event_totals", {})
    episodes = max(1, int(task_summary.get("episodes", 1)))
    chest_count = float(events.get("chest_opened", 0)) / episodes
    trap_count = float(events.get("trap_triggered", 0)) / episodes
    return (
        10000.0 * float(task_summary.get("success_rate", 0.0))
        + 500.0 * chest_count
        + 350.0 * float(milestones.get("key_collected", 0.0))
        + 350.0 * float(milestones.get("gold_collected", 0.0))
        + 250.0 * float(milestones.get("agent_healed", 0.0))
        + 250.0 * float(milestones.get("button_pressed", 0.0))
        + 250.0 * float(milestones.get("door_opened", 0.0))
        + 150.0 * float(milestones.get("monster_killed", 0.0))
        + float(task_summary.get("avg_reward", 0.0))
        - 300.0 * trap_count
    )


def run_case(case: SweepCase, args: argparse.Namespace, index: int) -> dict[str, Any]:
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{index:02d}_{case.name}.json"
    stdout_out = out_dir / f"{index:02d}_{case.name}.stdout"

    command = [
        sys.executable,
        "utils/evaluate_policy.py",
        "--policy",
        args.policy,
        "--tasks",
        *args.tasks,
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--json-out",
        str(json_out),
    ]
    env = os.environ.copy()
    env.update({key: str(value) for key, value in case.params.items()})

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_out.write_text(completed.stdout + completed.stderr, encoding="utf-8")

    payload: dict[str, Any] = {}
    if json_out.exists():
        payload = json.loads(json_out.read_text(encoding="utf-8"))

    task_summaries = payload.get("summary", {})
    score = sum(task_score(summary) for summary in task_summaries.values())
    return {
        "case": case.name,
        "params": case.params,
        "returncode": completed.returncode,
        "score": score,
        "json_out": str(json_out.relative_to(PROJECT_ROOT)),
        "stdout_out": str(stdout_out.relative_to(PROJECT_ROOT)),
        "summary": task_summaries,
    }


def main() -> None:
    args = parse_args()
    cases = CASES[: max(0, min(args.limit, len(CASES)))]
    results = [run_case(case, args, index) for index, case in enumerate(cases)]
    results.sort(key=lambda item: item["score"], reverse=True)

    out_dir = PROJECT_ROOT / args.out_dir
    summary_out = out_dir / "summary.json"
    summary_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote sweep summary to {summary_out.relative_to(PROJECT_ROOT)}")
    for result in results:
        task = result["summary"].get(TASK5_ID, {})
        print(
            f"{result['case']:24s} score={result['score']:.1f} "
            f"success={task.get('success_rate', 0):.3f} "
            f"reward={task.get('avg_reward', 0):.3f} "
            f"steps={task.get('avg_steps', 0):.1f} "
            f"params={result['params']}"
        )


if __name__ == "__main__":
    main()
