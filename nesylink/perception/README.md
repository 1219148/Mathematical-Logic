# Perception 接口说明

这个模块负责把一帧 RGB pixels 转成符号状态。推理阶段只允许使用图像，不读 `info`、`grid` 或环境内部状态。

## 推荐入口

统一使用 `PerceptionEngine.extract`：

```python
from nesylink.perception.engine import PerceptionEngine

engine = PerceptionEngine(device="cpu")
state = engine.extract(rgb_frame)
```

`rgb_frame` 要求是 `np.ndarray`，形状为 `H x W x 3`，RGB，`uint8`。如果输入高度大于地图高度，`engine` 会裁掉底部 UI 区域，只保留地图区域。

调试时可以这样接环境：

```python
from nesylink.env import make_env
from nesylink.perception.engine import PerceptionEngine

env = make_env(
    task_id="mathematical_logic/task_2",
    observation_mode="full",
    render_mode="rgb_array",
)
obs, info = env.reset(seed=0)

frame = env.render()
state = PerceptionEngine(device="cpu").extract(frame)
```

## 返回结构

`extract` 返回 `SymbolicState`，定义在 `nesylink/shared/types.py`。

常用字段：

```python
state.player            # 玩家 tile 坐标，例如 (7, 3)
state.walls             # 墙 tile 集合
state.monsters          # 怪物 tile 集合
state.chests            # 宝箱 tile 集合
state.traps             # 陷阱 tile 集合
state.exits             # 出口 tile 集合
state.buttons           # 按钮 tile 集合
state.gaps              # 缺口 tile 集合
state.bridges           # 桥 tile 集合
state.switches          # 开关 tile 集合
state.static_grid       # CNN 输出的完整 8x10 语义图

state.player_entity     # 玩家像素级实体信息
state.monster_entities  # 怪物像素级实体信息
```

`player_entity` 和 `monster_entities` 中的元素是 `EntityState`：

```python
entity.tile        # tile 坐标
entity.center_px   # 像素中心点 (x, y)
entity.bbox_px     # 16x16 bbox: (x0, y0, x1, y1)
entity.kind        # "player" 或 "monster"
entity.confidence  # CNN heatmap 置信度
```

推荐用法：

- 符号规划用 `state.player`、`state.walls`、`state.monsters` 等 tile 字段。
- 像素级执行、碰撞规避、距离判断用 `state.player_entity.center_px` 和 `state.monster_entities[*].bbox_px`。

## CNN 输出逻辑

模型在 `cnn.py` 中，主体是 `TinyPerceptionCNN`。它有两个 head：

- `tile_head`：输出 `8 x 10` 语义图，负责墙、宝箱、出口、陷阱、按钮、桥、缺口等 tile 级信息。
- `heatmap_head`：输出玩家/怪物的像素级中心点 heatmap，负责动态实体的精确位置。

所以 perception 不是只返回 tile，也不是只返回 pixel，而是混合表示：

```text
静态/地形/道具: tile 级
玩家/怪物动态实体: tile + pixel center + bbox
```

## 文件约定

```text
nesylink/perception/engine.py                 # 对外统一入口
nesylink/perception/cnn.py                    # CNN、数据采集、训练、评估
nesylink/perception/perception_model.pt       # 默认权重
nesylink/perception/data/perception_dataset.npz
nesylink/perception/data/generated_maps/      # 随机生成的训练地图
```

默认 `PerceptionEngine()` 会加载：

```text
nesylink/perception/perception_model.pt
```

如果要加载别的权重：

```python
engine = PerceptionEngine(
    weights_path="path/to/perception_model.pt",
    device="cpu",
)
```

## 训练和评估

所有命令在项目根目录运行：

```bash
cd /home/VIG/data2/dangyunkai/wuhaoyi/Mathematical-Logic
```

评估已有权重：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m nesylink.perception.cnn eval \
  --data nesylink/perception/data/perception_dataset.npz \
  --weights nesylink/perception/perception_model.pt \
  --device cpu
```

重新采集数据并训练：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m nesylink.perception.cnn collect-train \
  --samples 2400 \
  --epochs 18 \
  --batch-size 64 \
  --data nesylink/perception/data/perception_dataset.npz \
  --weights nesylink/perception/perception_model.pt \
  --device cpu
```

只采集数据：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m nesylink.perception.cnn collect \
  --samples 2400 \
  --output nesylink/perception/data/perception_dataset.npz
```

只训练已有数据：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m nesylink.perception.cnn train \
  --data nesylink/perception/data/perception_dataset.npz \
  --weights nesylink/perception/perception_model.pt \
  --epochs 18 \
  --batch-size 64 \
  --device cpu
```

## 当前权重指标

当前 `perception_model.pt` 在 `perception_dataset.npz` 上的评估结果：

```text
val_loss: 0.006514
tile_acc: 0.999807
player_center_error_px: 1.129
monster_tile_recall: 0.993506
```

单帧 CPU 推理耗时：

```text
首次调用，包含加载权重: 约 32 ms
模型已加载后，默认 64 CPU 线程: 约 8 ms / frame
16 CPU 线程: 约 6 ms / frame
```

如果用于 agent 循环，建议在程序开头限制线程数：

```python
import torch
torch.set_num_threads(16)
```

或在 shell 中：

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

## 注意事项

- 当前 perception 重点解决地图语义、玩家位置、怪物位置。
- `health`、`keys`、`gold`、`items` 目前没有从像素中恢复，`engine.extract` 中保守填默认值。
- `monster_types` 当前填 `"unknown"`，如需区分 chaser/patroller/ambusher，需要给 CNN 增加怪物类型分类 head 或引入时序跟踪。
- `static_grid` 字段名沿用现有接口，但内容是完整 CNN 语义图，里面也可能包含 player/monster 这类动态对象。
- 训练标签来自 structured observation，只在训练阶段使用；正式推理入口不要读这些标签。
