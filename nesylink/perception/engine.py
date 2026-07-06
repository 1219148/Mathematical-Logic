from __future__ import annotations
import numpy as np
from nesylink.shared import SymbolicState

class PerceptionEngine:
    def __init__(self) -> None:
        pass

    def reset(self, obs: np.ndarray) -> None:
        pass

    def extract(self, obs: np.ndarray) -> SymbolicState:
        raise NotImplementedError
