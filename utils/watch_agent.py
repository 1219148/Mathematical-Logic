"""Watch a submitted NesyLink policy play in a pygame window.

This is a debugging companion to ``utils/evaluate_policy.py``. It uses the same
policy loading path, but renders each step like the human-play tool so failures
can be inspected visually.

Usage:
    python utils/watch_agent.py --task mathematical_logic/task_5 --policy submissions/student_agent.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pygame
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pygame is required for utils/watch_agent.py. Install it with "
        "`python -m pip install -e '.[pygame]'`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_policy import call_policy, event_names, is_success, load_policy, reset_policy
from nesylink.core.constants import ACTION_LABELS, WINDOW_HEIGHT, WINDOW_WIDTH
from nesylink.env import make_env
from nesylink.tasks import list_tasks


OVERLAY_BG = (0, 0, 0, 170)
OVERLAY_TEXT = (250, 250, 250)
OVERLAY_DIM = (190, 200, 210)
OVERLAY_OK = (120, 255, 160)
OVERLAY_BAD = (255, 135, 135)


def parse_args() -> argparse.Namespace:
    task_ids = [task.task_id for task in list_tasks()]
    parser = argparse.ArgumentParser(description="Watch a NesyLink policy in a pygame window.")
    parser.add_argument("--policy", required=True, help="Policy module or file, optionally with :attribute.")
    parser.add_argument(
        "--task",
        default="mathematical_logic/task_5",
        choices=task_ids,
        help="Task ID to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override task max_steps.")
    parser.add_argument("--fps", type=int, default=30, help="Viewer frames per second.")
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Optional MP4/GIF output path. Requires imageio, for example `pip install imageio[ffmpeg]`.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="Video frame rate. Defaults to --fps.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Record without opening a visible pygame window. Useful for batch debugging.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Optional directory for saved PNG frames.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save one frame every N steps when --frame-dir is set.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Open the viewer paused. Press Space to run or N to step once.",
    )
    return parser.parse_args()


class VideoRecorder:
    def __init__(self, path: Path, fps: int) -> None:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "--video-out requires imageio. Install it with "
                "`.venv/bin/python -m pip install 'imageio[ffmpeg]'` for MP4 output."
            ) from exc

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=fps)

    def append_surface(self, surface: pygame.Surface) -> None:
        frame = pygame.surfarray.array3d(surface)
        self.writer.append_data(np.swapaxes(frame, 0, 1))

    def close(self) -> None:
        self.writer.close()


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape (H, W, 3), got {frame.shape}")
    return pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int], color) -> int:
    rendered = font.render(text, True, color)
    surface.blit(rendered, pos)
    return rendered.get_height() + 4


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    *,
    task_id: str,
    seed: int,
    step: int,
    action: int,
    reward: float,
    total_reward: float,
    done: bool,
    success: bool,
    terminal_reason: str | None,
    recent_events: list[str],
    event_counter: Counter[str],
    paused: bool,
) -> None:
    panel = pygame.Surface((WINDOW_WIDTH, 118), pygame.SRCALPHA)
    panel.fill(OVERLAY_BG)
    screen.blit(panel, (0, 0))

    action_label = ACTION_LABELS.get(action, str(action))
    status = "paused" if paused and not done else "running"
    if done:
        status = "success" if success else "ended"
    status_color = OVERLAY_OK if success else OVERLAY_BAD if done else OVERLAY_TEXT

    y = 8
    y += draw_text(screen, font, f"{task_id}  seed={seed}  step={step}", (10, y), OVERLAY_TEXT)
    y += draw_text(
        screen,
        font,
        f"action={action_label}  reward={reward:.3f}  total={total_reward:.3f}  status={status}",
        (10, y),
        status_color,
    )
    event_text = ", ".join(recent_events[-4:]) if recent_events else "-"
    y += draw_text(screen, font, f"events={event_text}", (10, y), OVERLAY_DIM)
    if terminal_reason:
        y += draw_text(screen, font, f"terminal={terminal_reason}", (10, y), OVERLAY_BAD)
    else:
        important = [
            f"{name}:{event_counter[name]}"
            for name in ("chest_opened", "key_collected", "agent_healed", "door_opened", "world_completed")
            if event_counter[name] > 0
        ]
        draw_text(screen, font, "counts=" + (", ".join(important) if important else "-"), (10, y), OVERLAY_DIM)

    help_text = "Space pause/run   N step   R reset   Esc quit"
    draw_text(screen, font, help_text, (10, WINDOW_HEIGHT - 24), OVERLAY_TEXT)


def save_frame(
    *,
    frame_dir: Path | None,
    step: int,
    save_every: int,
    screen: pygame.Surface,
) -> None:
    if frame_dir is None or save_every < 1 or step % save_every != 0:
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, str(frame_dir / f"step_{step:05d}.png"))


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    video_fps = args.video_fps if args.video_fps is not None else args.fps
    if video_fps < 1:
        raise ValueError("--video-fps must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    policy = load_policy(args.policy)
    env_kwargs: dict[str, Any] = {
        "observation_mode": "pixels",
        "render_mode": "rgb_array",
    }
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    env = make_env(task_id=args.task, **env_kwargs)

    pygame.init()
    pygame.display.set_caption(f"NesyLink Agent Viewer - {args.task}")
    if args.no_window:
        screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    else:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 16) or pygame.font.Font(None, 18)
    recorder = VideoRecorder(args.video_out, video_fps) if args.video_out is not None else None

    def reset_episode():
        reset_policy(policy, seed=args.seed, task_id=args.task)
        reset_obs, reset_info = env.reset(seed=args.seed)
        return {
            "obs": reset_obs,
            "info": reset_info,
            "reward": 0.0,
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
            "step": 0,
            "action": 0,
            "recent_events": [],
            "event_counter": Counter(),
            "success": False,
            "terminal_reason": None,
        }

    state = reset_episode()
    paused = bool(args.start_paused)
    step_once = False
    running = True
    last_captured_step: int | None = None

    try:
        while running:
            if not args.no_window:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_n:
                            step_once = True
                            paused = True
                        elif event.key == pygame.K_r:
                            state = reset_episode()
                            paused = bool(args.start_paused)
                            step_once = False
                            last_captured_step = None

            done = bool(state["terminated"] or state["truncated"])
            should_step = (args.no_window or not paused or step_once) and not done
            if should_step:
                action = call_policy(policy, state["obs"], state["info"])
                if not env.action_space.contains(action):
                    raise ValueError(f"policy returned invalid action {action!r}")
                obs, reward, terminated, truncated, info = env.step(action)
                names = event_names(info)
                state["obs"] = obs
                state["info"] = info
                state["reward"] = float(reward)
                state["total_reward"] += float(reward)
                state["terminated"] = bool(terminated)
                state["truncated"] = bool(truncated)
                state["step"] += 1
                state["action"] = action
                state["recent_events"] = names
                state["event_counter"].update(names)
                state["success"] = is_success(info, bool(terminated))
                state["terminal_reason"] = info.get("terminal_reason")
                step_once = False

            frame = env.render()
            raw_surface = frame_to_surface(frame)
            scaled = pygame.transform.scale(raw_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
            screen.blit(scaled, (0, 0))
            draw_overlay(
                screen,
                font,
                task_id=args.task,
                seed=args.seed,
                step=int(state["step"]),
                action=int(state["action"]),
                reward=float(state["reward"]),
                total_reward=float(state["total_reward"]),
                done=bool(state["terminated"] or state["truncated"]),
                success=bool(state["success"]),
                terminal_reason=state["terminal_reason"],
                recent_events=state["recent_events"],
                event_counter=state["event_counter"],
                paused=paused,
            )
            current_step = int(state["step"])
            if current_step != last_captured_step:
                save_frame(
                    frame_dir=args.frame_dir,
                    step=current_step,
                    save_every=int(args.save_every),
                    screen=screen,
                )
                if recorder is not None:
                    recorder.append_surface(screen)
                last_captured_step = current_step
            if not args.no_window:
                pygame.display.flip()
            elif state["terminated"] or state["truncated"]:
                running = False
            clock.tick(args.fps)
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
