from __future__ import annotations
import numpy as np

class Policy:
    def __init__(self) -> None:
        pass

    def reset(self, seed: int | None, task_id: str) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

def make_policy() -> Policy:
    return Policy()
