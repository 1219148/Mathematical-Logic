from __future__ import annotations

"""Default RL-agent entrypoint used by utils/evaluate_policy.py.

The previous tabular Q-learning implementation is kept in train_q_learning.py
and the saved model files. The default policy now uses the independent
hierarchical options agent, without importing submissions.student_agent.
"""

from .options_policy import OptionsRLPolicy, make_policy

Policy = OptionsRLPolicy

__all__ = ["OptionsRLPolicy", "Policy", "make_policy"]
