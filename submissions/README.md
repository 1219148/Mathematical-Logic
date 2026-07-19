# 官方测评仓库运行说明

将整个 `submissions/` 文件夹原样放到官方仓库
[`CrazyJassBread/nesylink`](https://github.com/CrazyJassBread/nesylink) 根目录后即可测评，
不需要复制本项目中的其他 Python 包。

例如：

```bash
python utils/evaluate_policy.py \
  --tasks mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_5=submissions/student_agent.py \
  --num-envs 1 \
  --info-mode safe
```

提交包自带：

- `shared.py`：感知与规划共用的 `EntityState`、`SymbolicState` 类型；
- `perception/`：CNN 代码、PyTorch 权重和 NumPy 兼容权重；
- `temporal_filter.py`、`student_agent.py`：正式策略；
- `rl_agent/`：可选 RL 实验策略和权重。

运行时只依赖官方环境公开的 `nesylink.core`、`nesylink.env` 等接口。提交代码不依赖
官方仓库中不存在的 `nesylink.shared` 或 `nesylink.perception`。

如果环境安装了 PyTorch，感知模块默认加载 `perception_model.pt`；如果没有安装
PyTorch，则自动加载同一 CNN 导出的 `perception_model.npz`，使用仅依赖 NumPy 的
推理后端。两条路径保持相同的 `PerceptionEngine` 和 policy 接口。
