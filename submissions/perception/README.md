# Perception 接口说明

这个模块负责把一帧 RGB pixels 转成符号状态。推理阶段只允许使用图像，不读 `info`、`grid` 或环境内部状态。

## 推荐入口

统一使用 `PerceptionEngine.extract`：

```python
from submissions.perception.engine import PerceptionEngine

engine = PerceptionEngine(device="cpu")
state = engine.extract(rgb_frame)
```

`rgb_frame` 要求是 `np.ndarray`，形状为 `H x W x 3`，RGB，`uint8`。如果输入高度大于地图高度，`engine` 会裁掉底部 UI 区域，只保留地图区域。

调试时可以这样接环境：

```python
from nesylink.env import make_env
from submissions.perception.engine import PerceptionEngine

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

`extract` 返回 `SymbolicState`，定义在提交包内的 `submissions/shared.py`，不依赖
官方仓库中不存在的 `nesylink.shared`。

常用字段：

```python
state.player            # 玩家 tile 坐标，例如 (7, 3)
state.walls             # 墙 tile 集合
state.monsters          # 怪物 tile 集合
state.chests            # 宝箱 tile 集合
state.traps             # 陷阱 tile 集合
state.exits             # 出口 tile 集合
state.exit_types        # 出口类型映射，例如 {(4, 0): "locked_key"}
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

模型在 `cnn.py` 中，主体是 `TinyPerceptionCNN`。它有五个 head：

- `tile_head`：输出 `8 x 10` 语义图，负责墙、宝箱、出口、陷阱、按钮、桥、缺口等 tile 级信息。
- `exit_type_head`：输出 `8 x 10` 出口类型图，负责区分 `normal`、`locked_key`、`conditional`。
- `chest_state_head`：输出独立的 `8 x 10` 宝箱状态图，负责区分 `none`、`closed`、`opened`；因此宝箱可以与桥等地形共用同一 tile。
- `heatmap_head`：输出玩家/怪物的像素级中心点 heatmap，负责动态实体的精确位置。
- `occupancy_head`：独立输出玩家/怪物的 `8 x 10` 占据概率，避免动态实体与桥、出口等
  共享 tile 时被单标签语义图覆盖。

所以 perception 不是只返回 tile，也不是只返回 pixel，而是混合表示：

```text
静态/地形/道具: tile 级
玩家/怪物动态实体: tile + pixel center + bbox
```

训练和评估显式覆盖 6 种颜色/亮度变体：

```text
default        原始 RGB 图像
grayscale      灰度图复制为 3 通道 RGB
dark           整体变暗
bright         整体变亮
high_contrast  灰度阈值二值化后的高对比图
inverted       RGB 反色
```

## 文件约定

```text
submissions/perception/engine.py                 # 对外统一入口
submissions/perception/cnn.py                    # CNN、数据采集、训练、评估
submissions/perception/perception_model.pt       # 默认权重
submissions/perception/numpy_inference.py         # 无 PyTorch 时的 NumPy CNN 后端
submissions/perception/perception_model.npz      # 同一 CNN 的 NumPy 兼容权重
submissions/perception/data/perception_dataset.npz
submissions/perception/data/generated_maps/      # 随机生成的训练地图
```

默认 `PerceptionEngine()` 会优先加载：

```text
submissions/perception/perception_model.pt
```

官方仓库的基础依赖没有声明 PyTorch。若运行环境未安装 PyTorch，入口会自动使用
`perception_model.npz` 和 `numpy_inference.py` 完成同一 CNN 的前向推理；无需修改
`PerceptionEngine`、`make_policy` 或 `act` 接口，也不使用颜色模板或关卡特判。

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
/home/VIG/data2/conda_envs/ml/bin/python -m submissions.perception.cnn eval \
  --data submissions/perception/data/perception_dataset.npz \
  --weights submissions/perception/perception_model.pt \
  --variant all \
  --device cpu
```

重新采集数据并训练：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m submissions.perception.cnn collect-train \
  --samples 2400 \
  --exit-overlap-ratio 0.15 \
  --epochs 16 \
  --batch-size 64 \
  --data submissions/perception/data/perception_dataset.npz \
  --weights submissions/perception/perception_model.pt \
  --chest-head-only \
  --device cpu
```

`--chest-head-only` 用于在已有兼容权重上冻结旧编码器、tile/出口/热力图 head，只训练独立宝箱状态 head。训练会先缓存 `8 x 10` 编码特征，减少 CPU 重复反向传播。

只采集数据：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m submissions.perception.cnn collect \
  --samples 2400 \
  --exit-overlap-ratio 0.15 \
  --output submissions/perception/data/perception_dataset.npz
```

只训练已有数据：

```bash
/home/VIG/data2/conda_envs/ml/bin/python -m submissions.perception.cnn train \
  --data submissions/perception/data/perception_dataset.npz \
  --weights submissions/perception/perception_model.pt \
  --epochs 16 \
  --batch-size 64 \
  --chest-head-only \
  --device cpu
```

## 当前权重指标

当前 `perception_model.pt` 的旧编码器、tile/出口/热力图 head 与上一版权重逐元素一致；新增宝箱状态 head 在独立随机种子、600 张图像的 holdout 上结果如下：

```text
variant        closed_recall  opened_recall  false_positive_rate
default        0.975225       0.984043       0.000382
grayscale      0.986486       0.992021       0.000382
dark           0.975225       0.992021       0.000382
bright         0.950450       0.992021       0.000382
high_contrast  1.000000       0.992021       0.000042
inverted       0.968468       0.978723       0.000403
```

以官方仓库 `CrazyJassBread/nesylink@036df78` 为基准，将 `submissions/` 单独复制到
仓库根目录，并在只安装官方 `pyproject.toml` 依赖（无 PyTorch、无
`nesylink.shared`）的全新虚拟环境中回归：Task 1 default 在 283 步完成，Task 5
default 在 1188 步完成，两者均触发 `world_completed`。另外抽样 60 帧对比 PyTorch
与 NumPy CNN，语义图、玩家/怪物 tile、宝箱状态和出口类型输出全部一致。

单帧 CPU 推理耗时：

```text
首次调用，包含加载权重: 约 32 ms
模型已加载后，默认 64 CPU 线程: 约 8 ms / frame
16 CPU 线程: 约 6 ms / frame
仅 NumPy 兼容后端: 约 0.2 s / frame（用于官方基础依赖下的可运行兜底）
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

- 当前 perception 重点解决地图语义、出口类型、玩家位置、怪物位置。
- `health`、`keys`、`gold`、`items` 目前没有从像素中恢复，`engine.extract` 中保守填默认值。
- `monster_types` 当前填 `"unknown"`，如需区分 chaser/patroller/ambusher，需要给 CNN 增加怪物类型分类 head 或引入时序跟踪。
- `static_grid` 字段名沿用现有接口，但内容是完整 CNN 语义图，里面也可能包含 player/monster 这类动态对象。
- `tile_head` 是单标签语义图；宝箱开闭状态由独立训练的 `chest_state_head` 提供，因此可与桥等地形同时存在，并可抑制已打开宝箱造成的假出口或假怪物。
- 训练标签来自 structured observation，只在训练阶段使用；正式推理入口不要读这些标签。
