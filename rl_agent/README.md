# RL Agent 说明

本目录实现的是独立于 `submissions/student_agent.py` 的强化学习方向 agent。默认入口仍是：

```bash
python utils/evaluate_policy.py --policy rl_agent.q_learning_agent --tasks mathematical_logic/task_1 mathematical_logic/task_2 mathematical_logic/task_3 mathematical_logic/task_4 mathematical_logic/task_5 --num-envs 5
```

## 当前结构

- `q_learning_agent.py`：评测入口，导出 `Policy` 和 `make_policy()`。
- `options_policy.py`：当前默认策略。它使用视觉/网格状态抽取，执行分层 option：开箱、战斗、按钮/开关、出口探索。
- `train_options_q.py`：高层 `Q(s, option)` 训练脚本，默认已经纳入 task1-task5。
- `models/options_q.pkl`：训练后的高层 option Q 表；可以通过环境变量 `NESYLINK_OPTIONS_MODEL` 指定或禁用。
- `pretrain_from_expert.py`：仅保留为实验脚本，默认 agent 不导入、不调用 teacher/student agent。

## 训练

默认训练 task1-task5：

```bash
python -m rl_agent.train_options_q --episodes 400
```

也可以指定任务子集：

```bash
python -m rl_agent.train_options_q --tasks mathematical_logic/task_1 mathematical_logic/task_5 --episodes 600
```

## 设计原则

当前策略不是复制教师轨迹，也没有在线兜底到 `student_agent`。底层移动使用 BFS/option controller，高层可以加载 Q 表选择 option；当某个状态没有可靠 Q 值时，使用通用探索规则兜底。

## 当前状态

- task1-task4：此前评测均可通过。
- task5：已经纳入训练脚本与验证范围；当前策略可以稳定完成起点箱子、按钮、南房间钥匙箱、东房间治疗箱和开门，但最后一个西房间箱子仍未稳定完成。

下一步建议继续优化 task5 的通用 option 学习，重点是出口定位、危险区避让和健康/步数压力下的高层 option 选择，而不是写 task5 专用路线。
