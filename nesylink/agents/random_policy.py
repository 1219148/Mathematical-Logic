from nesylink.agents import Policy
import numpy as np

class RandomPolicy(Policy):
    def __init__(self):
        # 初始化感知模块
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        #通过调用感知模块的extract函数得到state,基于state决策
        return np.random.randint(0, 7)

def make_policy() -> RandomPolicy:
    return RandomPolicy()