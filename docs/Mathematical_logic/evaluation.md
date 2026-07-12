# 数理逻辑任务测评说明

本文档以当前的 [测评脚本](../../utils/evaluate_policy.py) 为准，完整说明策略性能测评的成功判定、Agent 接口、输入信息、运行模式、观测变体和结果字段。

正式测评使用五个任务 `mathematical_logic/task_1` 到 `mathematical_logic/task_5`，并固定使用 `observation_mode="pixels"`。

## 一、评分依据

策略性能的主要指标是任务完成率：

```text
success_rate = 成功完成任务的 episode 数 / 总 episode 数
```

一次 episode 满足以下任一条件时记为成功：

- `info["game"]["world_completed"] == True`
- `info["terminal_reason"] == "world_completed"`
- episode 已终止，并且 reward 终止原因为 `world_completed`

因此，成功率按“是否完成整个任务”计算，不按单步动作是否正确计算。

测评脚本还会统计阶段性进展，例如：

- 物品搜集：`key_collected`、`gold_collected`、`item_collected`
- 宝箱与机关：`chest_opened`、`button_pressed`、`switch_activated`、`door_opened`
- 战斗与风险：`monster_killed`、`trap_triggered`、`agent_dead`
- 地图推进：`room_changed`、`exit_reached`、`environment_completed`

这些指标用于分析 Agent 已完成的子目标，不替代最终成功率。Task 3、Task 4 和 Task 5 还会输出对应的 milestone；Task 5 会额外打印主要游戏事件的累计次数。

## 二、提交 Policy

### 1. 共享策略

使用 `--policy` 为所有被测任务指定同一个策略：

```bash
python utils/evaluate_policy.py \
  --policy submissions/shared_agent.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2
```

共享策略在正式 `safe` 模式下不会收到任务编号，需要根据像素观测、奖励历史和物品栏自行决策。

### 2. 单任务专用策略

使用可重复的 `--task-policy TASK_ID=POLICY_SPEC` 为不同任务指定不同策略：

```bash
python utils/evaluate_policy.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2 \
  --task-policy mathematical_logic/task_1=submissions/task1_agent.py \
  --task-policy mathematical_logic/task_2=submissions/task2_agent.py
```

在正式 `safe` 模式下，通过 `--task-policy` 显式绑定的策略会在 `info["task_id"]` 中收到对应任务 ID。

如果同时提供 `--policy` 和 `--task-policy`，后者覆盖对应任务，其他任务继续使用共享策略。如果没有提供 `--policy`，则每个被测任务都必须有对应的 `--task-policy`。

### 3. Policy 文件接口

测评脚本按以下顺序从模块或 Python 文件中寻找策略对象：

1. `make_policy()`
2. `Policy` 类
3. 模块级 `policy` 对象
4. 模块级 `act` 函数

推荐接口：

```python
class Policy:
    def reset(self):
        pass

    def act(self, obs, info):
        return 0


def make_policy():
    return Policy()
```

每个 episode 开始前，如果策略实现了 `reset()`，脚本会无参数调用一次。脚本优先调用 `act(obs, info)`；仅实现单参数接口时，也支持 `act(obs)`。

动作返回值必须能转换为单个整数。脚本也接受 `{"action": value}`、非空 tuple/list 的第一个元素或 NumPy 标量，最终动作必须属于环境动作空间 `0..6`。

同一个 policy spec 当前只加载一次，其对象会在多个 episode 之间复用，`reset()` 只是新 episode 的回调。课程规则仍要求正式测评时只进行推理，不得借此跨 episode 更新模型或继续训练。

## 三、正式 Agent 输入

### 1. 像素观测

`obs` 是当前地图区域的 RGB 图像：

```text
shape: (128, 160, 3)
dtype: uint8
```

正式测评不会把结构化 observation 传给 Agent。

### 2. safe_info

正式测评使用默认的 `--info-mode safe`。共享策略收到：

```python
{
    "last_reward": 0.0,
    "inventory": {
        "gold": 0,
        "keys": 0,
        "items": ["sword", "shield"],
        "tools": ["sword", "shield"],
        "equipped": {"A": "sword", "B": "shield"},
    },
}
```

通过 `--task-policy` 绑定的专用策略额外收到：

```python
{"task_id": "mathematical_logic/task_3"}
```

`last_reward` 是上一步 `env.step()` 返回的标量奖励，在 reset 后第一次决策时为 `0.0`。

正式 `safe_info` 不包含玩家坐标、血量、朝向、房间 ID、对象数量、对象坐标、地图变体、颜色变体、seed、事件、完成状态、终止原因或 reward 细分信号。测评脚本内部仍使用完整环境信息统计结果，但不会通过 `safe_info` 传给 Agent。

### 3. full 调试模式

本地调试可以显式使用：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --tasks mathematical_logic/task_1 \
  --info-mode full
```

`full` 模式直接把环境原始 `info` 传给策略，不再使用上述 safe schema，可能包含内部状态、事件和 reward 信号。该模式只用于本地训练、oracle、可视化或排查问题，不属于正式测评口径。

## 四、固定比例鲁棒性测评

正式鲁棒性测评命令：

```bash
python utils/evaluate_policy.py \
  --policy submissions/student_policy.py \
  --info-mode safe \
  --robustness-suite \
  --num-envs 100 \
  --json-out results/robustness_suite_eval.json
```

五个任务分别使用专用策略时：

```bash
python utils/evaluate_policy.py \
  --tasks \
    mathematical_logic/task_1 \
    mathematical_logic/task_2 \
    mathematical_logic/task_3 \
    mathematical_logic/task_4 \
    mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_1=submissions/task1_agent.py \
  --task-policy mathematical_logic/task_2=submissions/task2_agent.py \
  --task-policy mathematical_logic/task_3=submissions/task3_agent.py \
  --task-policy mathematical_logic/task_4=submissions/task4_agent.py \
  --task-policy mathematical_logic/task_5=submissions/task5_agent.py \
  --info-mode safe \
  --robustness-suite \
  --num-envs 100 \
  --json-out results/robustness_suite_eval.json
```

也可以使用共享策略并覆盖少数任务：

```bash
python utils/evaluate_policy.py \
  --policy submissions/shared_agent.py \
  --task-policy mathematical_logic/task_4=submissions/task4_specialist.py \
  --task-policy mathematical_logic/task_5=submissions/task5_specialist.py \
  --info-mode safe \
  --robustness-suite \
  --num-envs 100
```

启用 `--robustness-suite` 后，`--num-envs` 表示每个 task 的总 episode 数。脚本按 60% / 30% / 10% 切分；当数量不能整除时，按最大余数法分配，三类 episode 总数仍等于 `--num-envs`。

| 阶段         | 比例 | `--num-envs 100` | 观测             | 地图                   |
| ------------ | ---: | -----------------: | ---------------- | ---------------------- |
| `original` |  60% |                 60 | `default`      | 原始地图               |
| `spatial`  |  30% |                 30 | `default`      | `spatial_a/b/c` 循环 |
| `color`    |  10% |                 10 | 五种颜色变体循环 | 原始地图               |

该固定套件适用于全部五个任务：

- `original` 使用原始地图和原始像素。
- `spatial` 使用临时生成的地图副本，改变部分布局、出生点或对象位置；三个固定地图变体循环使用。
- `color` 依次循环 `grayscale`、`dark`、`bright`、`high_contrast`、`inverted`，只改变传给策略的像素观测，不改变环境状态。

在鲁棒性模式下，episode 计划由脚本固定生成，命令行提供的 `--obs-variants` 不参与该计划。

## 五、episode 执行流程

每个 episode 的执行顺序为：

1. 选择当前任务绑定的 policy。
2. 无参数调用可选的 `policy.reset()`。
3. 使用当前 seed 调用 `env.reset()`。
4. 构造第一次 `safe_info`，其中 `last_reward=0.0`。
5. 循环调用 policy、检查动作、执行 `env.step()`，并将本步 reward 作为下一次决策的 `last_reward`。
6. episode 终止或达到任务 `max_steps` 后关闭环境并记录结果。

默认 episode seed 为 `--seed + episode_index`。`--max-steps` 和 `--action-repeat` 可以覆盖任务配置，但正式报告应记录是否使用了覆盖值。

## 六、结果解读

每个 episode 会输出一行结果，例如普通默认模式下：

```text
mathematical_logic/task_1 stage=default obs_variant=default map_variant=default seed=0 success=True steps=290 reward=127.050
```

summary 按 `(task_id, eval_stage)` 分组。鲁棒性模式下的示例：

```text
mathematical_logic/task_3 [spatial]
  episodes:     30
  success_rate: 0.700
  avg_steps:    420.5
  avg_reward:   18.300
  variants:     {'default': 30}
  map_variants: {'spatial_a': 10, 'spatial_b': 10, 'spatial_c': 10}
  milestones:
    monster_killed: 0.800
    key_collected: 0.700
  progress:
    monster_killed: 0.800
    key_collected: 0.700
```

主要字段含义：

| 字段             | 含义                                                      |
| ---------------- | --------------------------------------------------------- |
| `success_rate` | 该分组成功完成整个任务的 episode 比例                     |
| `avg_steps`    | 该分组平均执行步数                                        |
| `avg_reward`   | 该分组平均累计奖励                                        |
| `variants`     | 该分组包含的观测变体及数量                                |
| `map_variants` | 该分组包含的地图变体及数量                                |
| `milestones`   | 指定任务的关键事件达成率                                  |
| `progress`     | 某事件至少出现一次的 episode 比例；从未出现的事件不会显示 |

## 七、JSON 输出

传入 `--json-out PATH` 后，脚本写出：

```json
{
  "summary": {},
  "episodes": []
}
```

`summary` 与终端汇总一致；`episodes` 中每个元素包含：

| 字段                | 含义                       |
| ------------------- | -------------------------- |
| `task_id`         | 任务 ID                    |
| `eval_stage`      | 测评阶段                   |
| `obs_variant`     | 观测变体                   |
| `map_variant`     | 地图变体                   |
| `seed`            | episode 随机种子           |
| `steps`           | 执行步数                   |
| `total_reward`    | episode 累计奖励           |
| `terminated`      | 是否自然终止               |
| `truncated`       | 是否因步数上限截断         |
| `success`         | 是否完成整个任务           |
| `terminal_reason` | 终止原因                   |
| `event_counts`    | episode 内事件累计次数     |
| `milestones`      | 该任务的关键里程碑是否达成 |

`summary[分组]["progress_rates"]` 表示某事件至少出现一次的 episode 比例。例如：

```json
{
  "key_collected": 0.8,
  "chest_opened": 0.6,
  "monster_killed": 0.4
}
```

这表示 80% 的 episode 收集过钥匙、60% 打开过宝箱、40% 击杀过怪物，不表示这些 episode 已最终通关。

当前 JSON 不会自动记录完整命令、`info_mode`、policy 路径、`max_steps`、`action_repeat` 或代码版本，因此实验报告还应保存实际运行命令和仓库版本。

## 八、常用参数

| 参数                   | 默认值           | 含义                                                              |
| ---------------------- | ---------------- | ----------------------------------------------------------------- |
| `--policy`           | 无               | 共享 policy 文件或模块，可带`:attribute`                        |
| `--task-policy`      | 无               | `TASK_ID=POLICY_SPEC`，可重复并覆盖共享 policy                  |
| `--tasks`            | 五个数理逻辑任务 | 被测任务列表                                                      |
| `--num-envs`         | `100`          | 普通模式下为每 task、每观测变体的数量；鲁棒性模式下为每 task 总数 |
| `--seed`             | `0`            | episode seed 的起点                                               |
| `--max-steps`        | 任务配置         | 覆盖最大步数                                                      |
| `--action-repeat`    | 任务配置         | 覆盖动作重复次数                                                  |
| `--render-mode`      | `None`         | 可选`rgb_array`                                                 |
| `--info-mode`        | `safe`         | `safe` 用于正式测评，`full` 用于本地调试                      |
| `--obs-variants`     | `default`      | 普通模式使用的像素变体                                            |
| `--robustness-suite` | 关闭             | 启用固定 60% / 30% / 10% 套件                                     |
| `--json-out`         | 无               | 写出详细 JSON 结果                                                |

## 九、报告要求

正式实验报告至少应包含：

- 每个 task 在 `original`、`spatial`、`color` 三个阶段的成功率
- `avg_steps`、`avg_reward`、milestone 和 progress 指标
- 实际测评命令、seed、episode 数及是否覆盖 `max_steps` / `action_repeat`
- 使用的 policy 文件、模型权重和代码版本
- 训练或调试阶段是否使用过完整环境 `info`

正式成绩应以 `--info-mode safe --robustness-suite` 的实际运行结果为准。
