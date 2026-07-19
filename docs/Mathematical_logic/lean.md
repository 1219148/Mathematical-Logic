# Lean 形式化与证明

## 1. 环境形式化
#### 简化假设
为了便于形式化和后续证明，我们对游戏模型进行了以下假设。

**网格移动**

玩家和怪物只能在格子的中心移动。
为了建模怪物的速度只有玩家的一半，我们设计每回合玩家行动一次（向一个方向移动一格，向一格方向攻击/交互，防御），每两回合怪物行动一次。

**怪物模型简化**

怪物类型简化，假设所有的怪物都是chaser，向玩家移动。
怪物靠近玩家造成伤害后反弹，被玩家攻击后也会反弹，反弹之后忽略怪物晕眩的时间。
由于只有个别关卡的个别房间有两个怪物，忽略怪物重叠的情况。

**游戏砖块简化**

策略中并没有针对npc的逻辑，我们考虑将npc视作普通墙体。
陷阱类型简化，假设所有的陷阱都是spike，玩家踩到后扣血并回到当前房间重生点。

#### 基础类型（env1.lean）

**坐标：**

环境中的地图房间固定为 8 行 × 10 列的网格，每个格子由一个坐标唯一标识。在 Lean 中，我们用有限类型 Fin 来建模坐标：

```lean
def Coord := Fin 8 × Fin 10    
```

这里 Fin 8 表示 {0, 1, …, 7}，Fin 10 表示 {0, 1, …, 9}。使用 Fin 而非 Nat 的原因如下：
Nat 无法在类型层面排除越界坐标；每次访问地图都要携带 h : row < 8 ∧ col < 10 的证明;
而 Fin 8 和 Fin 10 类型本身保证了坐标合法——所有类型正确的 Coord 值都自动在 0…7 × 0…9 范围内。

**方向：**

游戏是二维的，因而只有上下左右四个方向，且每个时刻只能有一个方向。在 Lean 中，我们用一个枚举类型来建模方向：

```lean
inductive Direction where
  | up | down | left | right
```

**动作：**

游戏中只有四种动作：等待，移动（4个方向），攻击或交互（buttonA），防御（buttonB）。在 Lean 中，我们用一个枚举类型来建模动作：

```lean
inductive Action where
  | wait | move | buttonA | buttonB   
```

**输入结构：**

每一个时刻的输入由两个部分组成：动作和方向。我们规定：如果动作是移动，那么方向是玩家将要移动的方向；如果动作是另外三个，那么方向是当前玩家朝向。在 Lean 中，我们用一个结构体来建模输入：

```lean
structure Input where
  direction : Direction
  action : Action
```

**地图格子：**

在游戏中，一个地块有九种可能（地面、墙、陷阱、按钮、开关、宝箱、桥、门）。在 Lean 中，我们用一个枚举类型来建模地块：

```lean
inductive Tile where
  | ground                                            
  | wall                                             
  | spike                                             
  | button (pressed : Bool)                         
  | switch (state : Nat)                         
  | chest (opened : Bool) (content : Item)             
            (hidden : Bool) (cond : Condition)
  | bridge (switchRoom : Nat)                          
           (switchCoord : Coord) (activeState : Nat)
  | door (id : Nat)                                    
```

特别解释一下bridge的三个参数：switchRoom是控制当前桥的开关的房间号，switchCoord是开关在相应房间的坐标，activeState是一个状态值（例如在第四关中桥有三种状态，那么这个值就取0、1、2），当桥的状态值与相应开关的状态值相等时才能安全通过。
另外，我们对陷阱进行了简化，即认为所有陷阱都是spike（扣血并回到当前房间出生点），而不考虑更复杂的abyss陷阱。

**物品：**

宝箱中只有钥匙和剑这两种物品，盾牌是自带的。在 Lean 中，我们用一个结构体来建模物品：

```lean
inductive Item where
  | key
  | sword
  deriving DecidableEq, Repr
```

**怪物：**

由于游戏中的怪物攻击力和防御力都相同，也不存在任何正面或者负面增益效果，因此只需要记录怪物血量和位置就能表征一个怪物。在 Lean 中，我们用一个结构体来建模怪物：

```lean
structure Enemy where
  hp : Nat
  coord : Coord
```

我们对怪物进行了简化，即认为所有怪物都是chaser（始终向玩家移动），而不考虑更复杂的patroller和ambusher。

**玩家：**

玩家需要记录的信息有所处房间、在房间中的坐标、血量、金币数、钥匙数以及是否有剑。在 Lean 中，我们用一个结构体来建模玩家：

```lean
structure Player where
  room : Nat; coord : Coord; health : Nat
  gold : Nat; key : Nat; hasSword : Bool
```

**条件：**

在游戏中只存在三种条件（展示隐藏宝箱或者打开门的条件）：消耗钥匙、按下按钮或者击杀所有怪物。在 Lean 中，我们用一个枚举类型来建模条件：

```lean
inductive Condition where
  | None
  | consumeKey
  | ButtonPressed (buttonPos : Coord)
  | EnimyCleared
```

**门：**

对于门，需要记录门的编号，打开条件，是否打开，朝向，目标房间以及目标房间的目标坐标。在 Lean 中，我们用一个结构体来建模门：

```lean
structure DoorInfo where
  id : Nat
  condition : Condition
  isOpened : Bool
  orientation : Direction
  targetRoom : Nat
  targetCoord : Coord
```

**房间：**

对于一个房间，需要记录默认出生点来建模踩到陷阱的情况，还需要记录所有地块内容，所有门的信息，所有怪物信息，还有关于高度和宽度的不变量证明。在 Lean 中，我们用一个结构体来建模房间：

```lean
structure Room where
  spawn : Coord
  layout : List (List Tile)     
  doors : List DoorInfo
  enemies : List Enemy
  inv_height : layout.length = 8  
  inv_width  : ∀ row ∈ layout, row.length = 10  
```

**游戏状态：**

一个游戏状态由round（记录回合）、玩家信息和一个能由房间编号得到具体房间信息的函数组成。在 Lean 中，我们用一个结构体来建模游戏状态：

```lean
structure GameState where
  round : Nat
  player : Player
  rooms : Nat → Room              
```

这里之所以需要回合是因为考虑到游戏中玩家速度是怪物两倍，因此我们规定怪物只能在偶数回合行动，也就大致模拟了速度倍数的关系。

#### 状态转移（env2.lean）

**各种函数的作用：**

env2.lean 中的函数按功能分为四组，共同支撑最终的 step 归纳谓词。

*辅助函数：*
- `toFin8`：将自然数转化成 `Fin 8`
- `toFin10`：将自然数转化成 `Fin 10`
- `nth`：查找列表第 `n` 个元素
*获取游戏数据：*
- `front`：得到当前位置指定方向上的下一个位置
- `getRoom`：得到当前游戏状态下玩家所处房间
- `getTile`：得到指定房间指定位置的地块
- `isChest`：判断当前地块是否是宝箱
- `isSwitch`：判断当前地块是否是开关
- `findEnemy`：查找指定房间指定位置的怪物
- `findChest`：查找指定房间指定位置的宝箱
- `findSwitch`：查找指定房间指定位置的开关
- `findDoor`：查找指定房间指定编号的门
- `moveAway`：根据玩家位置和怪物位置判断怪物远离玩家的方向
- `moveTowardsPlayer`：根据玩家位置和怪物位置判断怪物靠近玩家的方向
- `near`：判断玩家和怪物是否相邻
- `isEnemyNear`：判断怪物是否邻近玩家
- `conditionSatisfied`：检查当前游戏状态下条件是否满足
- `orientationFit`：判断玩家朝向和门的朝向是否符合

*更新游戏数据：*
- `updateLayoutAt`：以指定规则更新房间布局
- `updateLayoutAt_length`：证明更新后房间长度满足要求
- `updateLayoutAt_width`：证明更新后房间宽度满足要求
- `moveEnemy`：将怪物向指定方向移动
- `killEnemy`：玩家在一个状态下攻击面向的方向的怪物，返回攻击后游戏状态
- `openChest`：玩家在一个状态下打开指定位置的宝箱，返回打开后游戏状态
- `toggleSwitch`：玩家在指定房间指定位置切换开关，返回切换后的房间信息
- `updatePlayer`：玩家通过了指定的门之后玩家的信息
- `pushButton`：玩家在指定房间指定位置按下按钮，返回按下后的房间信息
- `updateEnemy`：玩家所在房间怪物向玩家移动一步之后的房间信息
- `updateHealth`：根据玩家是否开盾以及房间信息和玩家位置判断玩家扣血情况
- `bound`：根据房间信息和玩家位置，将和玩家相邻的怪物反弹

*处理事件：*
- `handleWait`：处理等待
- `handleMove`：处理移动
- `handleInteract`：处理交互
- `handleDefense`：处理开盾

**单步转移关系：**

根据现有状态和输入进行状态转移，状态转移一步分为四种情况：等待、移动、攻击或交互、防御。上述四种是玩家的动作，而由于怪物只有移动这一种状态，因此我们在每一步执行前判断是否是偶数回合，若是则执行移动怪物、判断伤害、反弹造成伤害的怪物等等逻辑。在 Lean 中，我们用一个归纳谓词来建模单步状态转移：

```lean
inductive step : GameState → Input → GameState → Prop where
  | wait     (s : GameState) (d : Direction) : step s {direction := d, action := Action.wait} (handleWait { direction := d, action := Action.wait } s)
  | move     (s : GameState) (d : Direction) : step s {direction := d, action := Action.move} (handleMove { direction := d, action := Action.move } s d)
  | interact (s : GameState) (d : Direction) : step s {direction := d, action := Action.buttonA} (handleInteract { direction := d, action := Action.buttonA } s d)
  | defense  (s : GameState) (d : Direction) : step s {direction := d, action := Action.buttonB} (handleDefense { direction := d, action := Action.buttonB } s)
```

这是一个归纳谓词而非函数，每个构造子对应一种动作类型。：step 使用 Prop 而非 Bool/GameState 返回，是因为我们希望在其上做归纳推理。

另外，updateLayoutAt 函数附带了两个关于布局更新的正确性定理，用来证明按照一定方式更新后长宽都不变，仍然符合房间约束。这两个定理被 openChest、toggleSwitch、pushButton 等函数内部使用，用于构造新的 Room 结构体时填充 inv_height 和 inv_width 证明字段。

## 2. 策略形式化

### 2.1 s.lean 整体架构

`s.lean` 对 Python 中的 `student_agent.py` 进行了符号层面的抽象建模。按功能分为 6 层：

```
Layer 1: opaque 环境查询接口
Layer 2: AgentState 结构体
Layer 3: Objective / ObjectiveKind 目标类型
Layer 4: chooseBridgeObjective — 中心室/拉杆房决策
Layer 5: chooseLocalObjective — 顶层目标选择
Layer 6: executeObjective / actLocalPlanner — 动作执行
```

#### Layer 1: opaque 环境查询

所有与环境交互的查询函数均被声明为 `opaque`（不可展开），其行为仅通过公理约束：

| opaque 函数 | 对应 Python 行为 |
|-------------|-----------------|
| `getCurrentMonsters` | 从感知输出中提取怪物坐标列表 |
| `getCurrentChests` | 从感知输出中提取宝箱坐标列表 |
| `getCurrentExits` | 从感知输出中提取出口坐标列表 |
| `getCurrentSwitches` | 从感知输出中提取开关坐标列表 |
| `isAdjacent` | 判断两坐标是否相邻（曼哈顿距离 = 1） |
| `getDirectionTo` | 给定相邻坐标，返回朝向方向 |
| `nextStepTo` | BFS 寻路：从 start 到 goals（避开 blocked），返回下一步方向 |
| `hasBridge` / `getBridgeSides` | 桥/中心室特征判定 |
| `sideApproachGoals` | 计算从某侧接近出口的 approach tile 列表 |

**为什么使用 opaque？** (1) 这些函数在 Python 中由 BFS/A* 搜索或 CNN 感知实现，展开到 Lean 中将导致证明过于冗长且与具体搜索算法耦合；(2) opaque + 公理的方式将对具体实现的依赖转化为对行为规约的依赖，使得证明可以适配不同搜索策略或感知模型。

#### Layer 2: AgentState

```lean
structure AgentState where
  hubExplorationStarted   : Bool
  rememberedBlockers      : List Coord     -- 已交互/应屏蔽的坐标（开过的宝箱、杀过的怪物等）
  monsterObjectiveDone    : Bool
  sawMonsterObjective     : Bool
  monsterAbsenceTicks     : Nat
  currentTargetSide       : Option Direction
  switchHubSide           : Option Direction
  hubSwitchPositions      : List Coord
  pressedSwitchForTarget  : Bool
  exploredHubSides        : List Direction
  postGoalRotateBridge    : Bool
  postGoalSwitchPressed   : Bool
```

关键字段说明：`rememberedBlockers` 是安全性/活性证明的核心——它将已交互对象记录为"屏蔽坐标"，寻路时避开。`blockersCoverObstacles` 谓词断言所有墙和陷阱均在 blockers 中，这是定理 1~4 的前提条件。

#### Layer 3~4: 目标选择

```lean
inductive ObjectiveKind where
  | fight      -- 攻击怪物
  | interact   -- 交互（开宝箱/按按钮/切换开关）
  | navigate   -- 导航到目标坐标集
  | goExit     -- 前往某侧出口
  | idle       -- 空闲

structure Objective where
  kind : ObjectiveKind
  targets : List Coord
  side : Option Direction
  interactionKind : Option InteractionKind
```

`chooseLocalObjective` 的优先级逻辑（对应 Python 中的 `_choose_local_objective`）：

1. 有怪物且（有剑 或 非 Hub 探索模式）→ `fight`
2. 有有效宝箱（不在 blockers 中）→ `interact`
3. Hub 探索模式 → `chooseBridgeObjective`（处理桥/拉杆房逻辑）
4. 否则 → `goExit`（有可见出口侧）或 `navigate`（向出口坐标集导航）

**t1~t4 的 `chooseLocalObjective` 调用中 `useHubExploration = false`**，因此 `AgentState` 在执行 `executeObjective` 后保持不变。这是定理 3&4 能够将 `AgentState` 排除在归纳测度之外的关键前提。

#### Layer 5~6: 动作执行

`executeObjective` 将 `Objective` 翻译为 `Input × AgentState`。逻辑分支：

| ObjectiveKind | 玩家邻近目标？ | 输出 |
|---------------|---------------|------|
| `fight` | 是 | `getDirectionTo` → `ACTION_A`（攻击） |
| `fight` | 否 | `nextStepTo` → `ACTION_MOVE` |
| `interact` | 是 | `getDirectionTo` → `ACTION_A`（交互） |
| `interact` | 否 | `nextStepTo` → `ACTION_MOVE` |
| `goExit` | 在 approach tile 上 | 直接向 `side` 方向移动 |
| `goExit` | 不在 approach tile 上 | `nextStepTo` 到 approach tile |
| `navigate` | — | `nextStepTo` → `ACTION_MOVE` |
| `idle` | — | `WAIT` |

### 2.2 关键设计选择

#### 选择 1：t1~t4 统一策略 vs 每关单独

我们用单个 `strategyTasks1to4` 函数覆盖全部前四关，而非为每关写独立策略。这体现了"统一符号规划器"的思路，保证了策略的结构通用性。

#### 选择 2：AgentState 不变性

```lean
def strategyTasks1to4 (gs : GameState) (as : AgentState) : Input × AgentState :=
  let (obj, as') := chooseLocalObjective gs as false
  executeObjective gs as' obj
```

通过 `chooseLocalObj_snd_eq` 和 `executeObjective_snd_eq` 公理，可证明：

```lean
theorem strategy_snd_eq (gs : GameState) (as : AgentState) :
    (strategyTasks1to4 gs as).snd = as := ...
```

**推论**：在 t1~t4 中，`AgentState` 在每一步执行后与执行前完全相同。这直接简化了 `steps` 关系的定义（只需一个 `as` 参数在每一步重复使用）和活性证明的归纳假设（只需对 `GameState` 做归纳，`as` 作为常量）。

#### 选择 3：`steps` 多步执行关系

```lean
inductive steps : GameState → AgentState → Nat → GameState → AgentState → Prop where
  | refl  (gs : GameState) (as : AgentState) : steps gs as 0 gs as
  | succ  (gs : GameState) (as : AgentState) (gsMid : GameState)
      (gs' : GameState) (as' : AgentState) (n : Nat) :
      step gs (strategyTasks1to4 gs as).fst gsMid →
      steps gsMid as n gs' as' →
      steps gs as (n + 1) gs' as'
```

与标准的 reflexive-transitive closure 不同，这里：

- 每一步都使用 `strategyTasks1to4` 决策（而非任意的 `Input`），因为我们要证明的是"策略+环境闭环系统"的性质，而非环境本身的任意迹
- `succ` 构造子中 `steps gsMid as n gs' as'` 的 `as` 不变——利用策略不修改 AgentState 的性质
- 步数 `n` 是显式的自然数参数，使强归纳法能够按步数/测度进行

---

## 3. 定理与证明

### 3.1 定理概览

| 定理 | 类型 | 直观含义 | 形式陈述 |
|------|------|----------|----------|
| **定理 1** `no_wall_bump` | 安全性 (Safety) | 策略输出 move 时，前方不是墙 | `getTile … (front …) ≠ Tile.wall` |
| **定理 2** `no_spike_step` | 安全性 (Safety) | 策略输出 move 时，前方不是陷阱 | `getTile … (front …) ≠ Tile.spike` |
| **定理 3** `single_monster_kill` | 活性 (Liveness) | 单怪物场景下，有限步内怪物必被消灭 | `∃ n gs' as', steps gs0 as0 n gs' as' ∧ enemies = []` |
| **定理 4** `single_chest_open` | 活性 (Liveness) | 无怪物、有有效宝箱场景下，有限步内宝箱全开 | `∃ n gs' as', steps … ∧ validChests = []` |

### 3.2 定理 1 & 2：安全性证明

#### 前提条件

两个定理共享同一个前置条件 `blockersCoverObstacles gs as.rememberedBlockers`，即：

```lean
def blockersCoverObstacles (gs : GameState) (blockers : List Coord) : Prop :=
  ∀ (c : Coord),
    (let t := getTile (getRoom gs) c; t = Tile.wall ∨ t = Tile.spike) →
    c ∈ blockers
```

这意味着 `rememberedBlockers` 包含了**所有**墙和陷阱坐标。结合 `nextStepTo_avoids_blocked` 公理——"`nextStepTo` 返回的方向不会走向 blockers 中的坐标"——即可推出安全性。

#### 证明流程

对 `Objective.kind` 进行**五项分类讨论**：

**Step 1**：展开策略到 `executeObjective` 调用层

```lean
rw [strategy_expand gs as] at hMove ⊢
generalize hPair : chooseLocalObjective gs as false = pair
rcases pair with ⟨obj, as'⟩
```

**Step 2**：证明 `as'.rememberedBlockers = as.rememberedBlockers`（利用 `chooseLocalObj_snd_eq` 公理），将 `hCovers` 迁移到 `hCovers' : blockersCoverObstacles gs as'.rememberedBlockers`。

**Step 3**：对 `obj.kind` 分类：

| `obj.kind` | `action = move` 何时出现？ | 安全论证 |
|------------|---------------------------|----------|
| `idle` | 永不出 move | 前提矛盾，自动消去 |
| `fight` | 不邻近任何怪物 → `nextStepTo` 返回移动方向 | `nextStepTo_avoids_blocked` 保证目标格 ∉ blockers，而 blockers 包含所有墙/陷阱 |
| `interact` | 不邻近任何交互对象 → `nextStepTo` 返回移动方向 | 同上 |
| `goExit` | 在 approach tile 上 → 向出口方向移动；否则 → `nextStepTo` 到 approach tile | 前者用 `exit_direction_not_wall`/`exit_direction_not_spike`；后者同上 |
| `navigate` | `nextStepTo` 返回移动方向 | `nextStepTo_avoids_blocked` |

**定理 1（不撞墙）**：
- 在 `nextStepTo_avoids_blocked` 分支中：`hAvoid : ¬ (front start d ∈ blocked)`，引入 `hWall : ... = Tile.wall`，则 `apply hAvoid; apply hCovers'; left; exact hWall` → 矛盾
- 在 `goExit` 的 approach 分支中：直接调用 `exit_direction_not_wall`

**定理 2（不踩陷阱）**：
- 结构完全对称，区别仅在于 `nextStepTo_avoids_blocked` 分支用 `hCovers'; right; exact hSpike`
- 在 `goExit` 的 approach 分支中：调用 `exit_direction_not_spike`
- `fight` 分支使用 `generalize hFilter` 同步 `head?` 结果，避免 Lean 将 `head?` 展开导致无法匹配

#### 定理 2 的特殊技巧：`generalize`

`fight` 分支中 `executeObjective` 内部有 `(obj.targets.filter (fun m => isAdjacent ...)).head?`。如果直接在 `hMove` 中展开，后续 `match` 分支的类型会依赖于具体的 `head?` 结果（`some m` / `none`），且不同分支在 Lean 眼中是不同类型，无法统一处理。因此使用：

```lean
generalize hFilter : (obj.targets.filter (fun m => isAdjacent gs.player.coord m)).head? = filterResult
```

将 `head?` 结果抽象为变量 `filterResult`，在 `filterResult` 的不同值上分别处理，保持各分支类型一致。

### 3.3 定理 3 & 4：活性证明（核心亮点，应重点阐述）

#### 3.3.1 总体架构：强归纳法 + 终止测度

两个活性定理都采用**良基归纳法**（well-founded induction）——具体是自然数上的强归纳法 `Nat.strongRecOn`：

```
目标：证明对所有 GameState，存在有限步 n 达成目标状态
方法：对"终止测度" m 做强归纳
      - 归纳假设：对所有测度 < m 的状态，性质成立
      - 归纳步骤：证明对测度 = m 的状态，经一步后测度严格减小（由核心公理保证）
                    → 落到归纳假设覆盖范围 → 存在有限步
```

这与编程语言中证明程序终止的典型方法一致：构造一个严格递减且下有界的测度函数。

#### 3.3.2 测度设计

**定理 3（怪物击杀）的测度：**

```lean
def monsterMeasure (gs : GameState) : Nat :=
  totalMonsterHP gs * 100 + minCoordDist gs.player.coord (getCurrentMonsters gs)
```

| 分量 | 含义 | 变化趋势 |
|------|------|----------|
| `totalMonsterHP gs` | 当前房间所有怪物 HP 之和 | 攻击命中时至少 -1 |
| `minCoordDist ...` | 玩家到最近怪物的曼哈顿距离 | 向怪物移动时缩小 |

**测度设计原理：** 乘数 `100` 是关键——它远大于网格上任意两点之间的曼哈顿距离（8×10 网格中最大距离 = 7 + 9 = 16）。这保证了测度在**字典序**上的行为：
- 如果 HP 减少 → 测度减少 ≥ 100，无论距离如何变化
- 如果 HP 不变 → 测度完全由距离决定，BFS 移动每步使距离严格减小

这使得测度在每一步要么因 HP 减少而大幅下降，要么因靠近怪物而小幅下降——二者结合保证严格单调递减。

**定理 4（宝箱打开）的测度：**

```lean
def chestMeasure (gs : GameState) (blockers : List Coord) : Nat :=
  let validChests := (getCurrentChests gs).filter (fun c => !blockers.contains c)
  validChests.length * 100 + minCoordDist gs.player.coord validChests
```

| 分量 | 含义 | 变化趋势 |
|------|------|----------|
| `validChests.length` | 未被 blocker 屏蔽的宝箱数 | 开箱成功时 -1 |
| `minCoordDist ...` | 玩家到最近有效宝箱的距离 | 向宝箱移动时缩小 |

与定理 3 同样的乘数 100 设计——打开宝箱使有效宝箱数 -1，测度减少 ≥ 100，主导距离的任何可能波动。

#### 3.3.3 定理 3 的详细证明流程

**前置条件：**

```lean
theorem single_monster_kill (gs0 : GameState) (as0 : AgentState)
    (hMonsterCount : (getCurrentMonsters gs0).length = 1)    -- 恰好一只怪物
    (hHasSword : gs0.player.hasSword = true)                  -- 玩家有剑
    (hCover : blockersCoverObstacles gs0 as0.rememberedBlockers) :  -- blockers 覆盖障碍物
    ∃ (n : Nat) (gs' : GameState) (as' : AgentState),
      steps gs0 as0 n gs' as' ∧ (getRoom gs').enemies = []
```

**证明步骤：**

**(a) 推导 `isEmpty = false`：**
从 `length = 1` → `getCurrentMonsters gs0 ≠ []` → `isEmpty = false`。

**(b) 定义归纳性质 P：**

```lean
let P : Nat → Prop := λ n => ∀ (gs : GameState) (as : AgentState),
  monsterMeasure gs = n →                    -- 测度 = n
  (getCurrentMonsters gs).isEmpty = false →  -- 有怪物
  gs.player.hasSword = true →               -- 有剑
  blockersCoverObstacles gs as.rememberedBlockers →  -- 障碍覆盖
  ∃ (n' : Nat) (gs' : GameState) (as' : AgentState),
    steps gs as n' gs' as' ∧ (getRoom gs').enemies = []
```

`P n` 说的是：**对所有测度等于 n 且满足前置条件的状态，存在有限步使怪物被消灭**。

**(c) 归纳步骤 `hStep : ∀ n, (∀ m < n, P m) → P n`：**

1. **判断 enemies 是否已空**：若 `enemies = []`，返回 `n' = 0`（零步，`steps.refl`）

2. **从核心公理获取一步进展**：
   ```lean
   have hProgress : ∃ gs', step gs (strategyTasks1to4 gs as).fst gs' ∧
       monsterMeasure gs' < monsterMeasure gs :=
     not_not_elim (strategy_fight_step_decreases_measure gs as hMonstersNE hSword hCoverArg)
   ```
   得到中间状态 `gsMid`、转移 `hStepMid`、测度减小 `hMeasureLt`。

3. **重建归纳假设的前置条件**：
   - `hSwordMid`：由 `hasSword_preserved_step` 公理保证 `gsMid.player.hasSword = true`
   - `hCoverMid`：由 `blockersCoverObstacles_preserved` 公理保证 blockers 覆盖在 `gsMid` 中仍成立

4. **判断 gsMid 中怪物状态**：
   - **仍有怪物**：`hm : monsterMeasure gsMid < n`（由 `hMeasureLt` 和 `hMeasure`），使用归纳假设 `hn (monsterMeasure gsMid) hm gsMid as rfl hMonstersNEMid hSwordMid hCoverMid`，得到 `n'` 步后的结果，加上当前这一步 → 返回 `n' + 1`
   - **怪物已消灭**：返回 `n' = 1`（一步达成）

**(d) 启动归纳：**

```lean
have hAll : ∀ n, P n := λ n => Nat.strongRecOn n hStep
exact hAll (monsterMeasure gs0) gs0 as0 rfl hMonstersNE0 hHasSword hCover
```

#### 3.3.4 定理 4 的详细证明流程

定理 4 的整体结构与定理 3 类似，但有一些值得报告的差异：

**差异 1：归纳性质仅参数化 GameState**

```lean
let blockers := as0.rememberedBlockers   -- 常量！策略不修改 as
let P : Nat → Prop := λ n => ∀ (gs : GameState),
  chestMeasure gs blockers = n →
  (getCurrentMonsters gs).isEmpty = true →
  gs.player.key = 0 →
  blockersCoverObstacles gs blockers →
  (getCurrentChests gs).filter (fun c => !blockers.contains c) ≠ [] →
  ∃ (n' : Nat) (gs' : GameState),
    steps gs as0 n' gs' as0 ∧    -- as0 直接出现在结论中
    (getCurrentChests gs').filter (fun c => !blockers.contains c) = []
```

因为 t1~t4 中策略不修改 `AgentState`，`as0` 自始至终不变。`blockers` 定义为 `as0.rememberedBlockers`，在全部归纳步骤中保持常量。

**差异 2：目标条件是有效宝箱集为空**

定理 4 的目标不是"所有宝箱都是 opened 状态"，而是"有效宝箱集（不在 blockers 中的未开宝箱）为空"。这与 `chestMeasure` 的定义一致：`validChests.length * 100 + ...`。

**差异 3：需要更多不变量的保持**

除了 `blockersCoverObstacles_preserved` 之外，定理 4 的归纳步骤还需：
- `noMonsters_preserved_step`：无怪物状态在 step 中保持（怪物不会凭空产生）
- `key_stays_zero_while_chests`：有有效宝箱等待开启时，key 保持为 0（宝箱内容不产生钥匙，或系统的钥匙不会改变有效宝箱集）

`key_stays_zero_while_chests` 公理是定理 4 的一个**关键简化假设**——它排除了"宝箱含钥匙 → key > 0 → 行为改变"的复杂情况。

**差异 4：最终结果组装**

因为 `P n` 返回 `∃ gs', steps gs as0 n' gs' as0 ∧ ...`（只量化了 `gs'`，不含 `as'`），而定理陈述要求 `∃ n gs' as', ...`，最终用 `rcases` + `exact ⟨n, gs', as0, hSteps, hAllOpened⟩` 组装。

#### 3.3.5 `not_not_elim` 辅助引理

由于 Lean 4.29-rc6 解析器 bug，所有公理的返回类型包装在 `¬ (¬ ...)` 中。证明中需要用 `not_not_elim` 提取正向陈述：

```lean
private theorem not_not_elim {P : Prop} (h : ¬ (¬ P)) : P := by
  by_cases hP : P
  · exact hP
  · exfalso; exact h hP
```

使用方式：
```lean
have hPreservedWrapped := hasSword_preserved_step gs gsMid ... hStepMid
have hPreserved : gsMid.player.hasSword = gs.player.hasSword :=
  not_not_elim hPreservedWrapped
```

这是一种**工程折中**——理想情况下公理应直接陈述正向命题，但受限于工具链的解析器 bug 而采用双重否定包装。

---

## 4. 公理体系

### 4.1 公理分类与完整清单

#### 共享辅助定义（4 个，非公理）

| 定义 | 签名 | 用途 |
|------|------|------|
| `coordDist` | `Coord → Coord → Nat` | 曼哈顿距离 |
| `minCoordDist` | `Coord → List Coord → Nat` | 到坐标集的最小曼哈顿距离 |
| `totalMonsterHP` | `GameState → Nat` | 当前房间所有怪物 HP 之和 |
| `blockersCoverObstacles` | `GameState → List Coord → Prop` | blockers 是否包含所有墙和陷阱 |

#### 安全性公理（5 条，定理 1 & 2）

| 公理 | 完整签名 | 角色 |
|------|----------|------|
| `nextStepTo_avoids_blocked` | `nextStepTo start goals blocked = some d → ¬ (front start d ∈ blocked)` | 寻路不走向 blocked 坐标 — 安全性的直接来源 |
| `exit_direction_not_wall` | `sideApproachGoals gs side blocked 包含 pCoord → ¬ (getTile … (front pCoord side) = Tile.wall)` | 出口 approach tile 向出口方向移动时不撞墙 |
| `exit_direction_not_spike` | 同上，结论为 `≠ Tile.spike` | 同上，不踩陷阱 |
| `getDirectionTo_correct` | `getDirectionTo src dst = some d → ¬ (¬ (front src d = dst))` | 方向计算正确性 — 面向目标 |
| `isAdjacent_correct` | `¬ (¬ (isAdjacent c1 c2 = true ↔ near c1 c2))` | `isAdjacent` 与 `near`（曼哈顿距离=1）等价 |

#### 活性公理（17 条，定理 3 & 4）

**寻路相关（3 条）：**

| 公理 | 含义 |
|------|------|
| `nextStepTo_some` | 目标集非空且不邻近任何目标时，`nextStepTo` 总能返回方向（BFS 完备性） |
| `nextStepTo_progress` | `nextStepTo` 返回的方向使玩家到目标集的最小曼哈顿距离严格减小（BFS 正确性） |
| `getDirectionTo_adjacent` | 两坐标相邻时，`getDirectionTo` 总能返回方向 |

**环境状态查询（2 条）：**

| 公理 | 含义 |
|------|------|
| `getCurrentMonsters_empty_iff` | `getCurrentMonsters` 为空 ↔ 房间 enemies 为空 |
| `getCurrentMonsters_mem` | 坐标在 `getCurrentMonsters` 中 ↔ 该坐标有怪物 |

**状态转移语义（3 条）：**

| 公理 | 含义 |
|------|------|
| `killEnemy_totalHP_lt` | 有剑且前方有怪物时，`killEnemy` 使总 HP 严格减小 |
| `handleMove_player_coord` | 前方非墙非刺时，移动后玩家坐标 = 前方坐标 |
| `handleMove_totalHP_nonincrease` | 移动操作不增加怪物总 HP（怪物不会在移动中回血/产生） |

**宝箱语义（2 条）：**

| 公理 | 含义 |
|------|------|
| `openChest_opens` | `openChest` 将未开启宝箱 (`chest false ...`) 变为已开启 (`chest true ...`) |
| `getCurrentChests_excludes_opened` | `getCurrentChests` 排除已开启 (`chest true ...`) 的宝箱 |

**状态不变性（5 条）：**

| 公理 | 保持什么 | 为什么需要 |
|------|----------|------------|
| `hasSword_preserved_step` | `hasSword` | 定理 3 归纳步骤需保证有剑状态不会丢失 |
| `blockersCoverObstacles_preserved` | `blockersCoverObstacles` | 定理 1~4 归纳步骤均需保证障碍覆盖性质不变 |
| `noMonsters_preserved_step` | `noMonsters` | 定理 4 归纳步骤需保证在宝箱场景中不会出现新怪物 |
| `key_stays_zero_while_chests` | `key = 0` | 定理 4 归纳步骤需保证 key 不变成正数（避免触发新行为分支） |
| `executeObjective_snd_eq` | `AgentState` | 定理 1&2 `hAs'` 推导 + 定理 3&4 `as` 不变性 |

**核心活性公理（2 条）：**

| 公理 | 完整含义 |
|------|----------|
| `strategy_fight_step_decreases_measure` | 有怪物 + 有剑 + 障碍覆盖 → 存在一步转移使 `monsterMeasure` 严格减小 |
| `strategy_interact_step_progress` | 无怪物 + 有有效宝箱 + key=0 + 障碍覆盖 → 存在一步转移使 `chestMeasure` 严格减小 |

这两条是最关键的**活性公理**，封装了"策略+环境"闭环的单步进展保证。它们综合了寻路进展（`nextStepTo_progress`）、攻击效果（`killEnemy_totalHP_lt`）、开箱效果（`openChest_opens`）、移动正确性（`handleMove_player_coord`）等底层公理的组合效应。

### 4.2 公理的合理性论证框架

课程要求"不允许未说明的 `axiom`"。下面从四个角度论证每条公理的合理性：

#### 角度 1：对应 Python 实现

每条公理可以关联到 Python 代码中的具体行为：

| 公理 | Python 对应 |
|------|------------|
| `nextStepTo_avoids_blocked` | `_bfs` 返回的路径第一步不会走向 `blocked` 参数中标记的坐标 |
| `nextStepTo_progress` | BFS 每步扩展使 frontier 到目标的距离缩短 |
| `nextStepTo_some` | 只要目标集非空且玩家不在目标上，BFS 总能找到一条路径（网格连通前提） |
| `killEnemy_totalHP_lt` | `killEnemy` 中 `hp - 1` 或 `filter (en => en.coord != coord)` 使 HP 总和减少 |
| `handleMove_player_coord` | `handleMove` 的 `Tile.ground` / `Tile.switch` / `Tile.door` 分支设置 `coord := front ...` |
| `openChest_opens` | `openChest` 中 `updateLayoutAt` 将 `Tile.chest false ...` 替换为 `Tile.chest true ...` |

#### 角度 2：环境语义保证

有些公理描述的是环境本身的不变性质，不依赖策略：

- `handleMove_totalHP_nonincrease`：移动操作不会创造怪物或给怪物回血，这是模拟器语义的内在性质
- `noMonsters_preserved_step`：怪物不会凭空产生（只在房间初始化时存在），这也是环境的语义保证
- `hasSword_preserved_step` / `blockersCoverObstacles_preserved`：这些是环境状态的自然不变量

#### 角度 3：策略行为保证

有些公理描述策略内部调用的函数的正确性——这些函数在 Python 中有确定的实现，但 Lean 中因声明为 `opaque` 而需要公理：

- `executeObjective_snd_eq`：从 `executeObjective` 的代码可见，每个分支返回的 `as` 都是传入的 `as`（`({direction := ..., action := ...}, as)`），从未修改
- `getDirectionTo_adjacent`：`getDirectionTo` 在两坐标相邻时只需比较坐标差即可确定方向

#### 角度 4：可独立验证

每条公理描述的行为都可以在 Python 环境中通过单元测试验证：

- 设置特定 GameState → 调用对应函数 → 检查返回值或状态变化是否符合公理断言
- 这将形式化证明与"代码确实运行"的可信性连接起来

### 4.3 公理与定理的依赖关系

```
安全性公理 (nextStepTo_avoids_blocked,
            exit_direction_not_wall/spike,
            getDirectionTo_correct)
    └─→ 定理1 (no_wall_bump)
    └─→ 定理2 (no_spike_step)

底层寻路公理 (nextStepTo_some, nextStepTo_progress, getDirectionTo_adjacent)
底层状态公理 (killEnemy_totalHP_lt, handleMove_player_coord,
            handleMove_totalHP_nonincrease, openChest_opens, ...)
    └─→ 核心活性公理 (strategy_fight_step_decreases_measure,
                      strategy_interact_step_progress)
            └─→ 定理3 (single_monster_kill)  ← + hasSword_preserved_step,
                                              |   blockersCoverObstacles_preserved
            └─→ 定理4 (single_chest_open)    ← + noMonsters_preserved_step,
                                                  key_stays_zero_while_chests
```

---

## 5. 形式化范围与局限性

### 已覆盖

- `Coord`、`Direction`、`Action`、`Tile`、`Enemy`、`Player`、`Room`、`GameState` 等**核心类型的完整建模**
- 四种动作（`move` / `interact` / `wait` / `defense`）的**状态转移关系**（`step` 归纳谓词）
- 基于 `blockers` 的**障碍物屏蔽机制**（将已开宝箱/已杀怪物等从寻路目标中排除）
- **安全性**：策略输出 `move` 时前方如果是墙/陷阱则与公理矛盾 → 一定不是墙/陷阱
- **活性**：单怪物 + 有剑 → 有限步消灭；无怪物 + 有宝箱 + key=0 → 有限步全开

### 未覆盖 / 有意的简化

| 简化项 | 当前假设 | 实际 Python 环境 | 影响 |
|--------|----------|-----------------|------|
| **多怪物场景** | 定理 3 假设 `length = 1` | t4/t5 中存在多怪物 | 多怪物收敛性需要更复杂的测度（如 `∑ HP_i × 100 + minDist_i` 的字典序） |
| **宝箱内容多样性** | 定理 4 假设宝箱不产生钥匙（`key_stays_zero_while_chests`） | t3/t4 中钥匙宝箱 → key > 0 后行为改变 | 需要分阶段证明：先拿钥匙 → 用钥匙开门 → 再处理其他宝箱 |
| **跨房间导航** | 定理 3&4 仅涉及单房间内的性质 | t3~t5 涉及多房间移动 | `room_memory` / `exit_memory` / 房间签名匹配等判定逻辑未被形式化 |
| **桥/按钮/开关机制** | 未形式化 | t4/t5 的核心机制 | `chooseBridgeObjective` 的复杂状态机逻辑未被证明 |
| **怪物 AI 行为** | 怪物同质化（不区分 chaser/patroller/ambusher） | Python 中三种 AI 有所不同 | 不区分AI不影响安全性证明（因为怪物移动不改变墙/陷阱布局），但可能影响活性测度的单调性 |
| **像素级移动** | Lean 中 `Action.move` = tile 级移动（一次一格） | Python 中是像素级（16 帧同方向 = 1 tile） | 有意的抽象层简化——tile 级建模足以刻画所有安全性和目标可达性 |
| **感知不确定性** | Lean 中 `GameState` 完全已知、可直接访问 | Python 策略只能通过 `obs` 图像帧推断 | Lean 证明"给定完美感知"下的性质；Python 实际表现另需感知模块准确性保障 |

### 多层验证架构

本次 Lean 形式化验证在整个验证体系中的定位如下：

```
┌─────────────────────────────────────────────┐
│ Python 层：端到端执行                         │
│  obs(像素) → PerceptionEngine → 符号状态      │
│  → choose_objective → BFS/A* → action        │
│  验证方式：鲁棒性测评 (--robustness-suite)     │
├─────────────────────────────────────────────┤
│ Lean 层：符号逻辑验证                         │
│  GameState(完美已知) → strategyTasks1to4      │
│  → step → GameState'                         │
│  验证方式：安全性定理 + 活性定理              │
│  前提：感知模块输出正确（准确提取符号状态）     │
└─────────────────────────────────────────────┘
```

两层验证之间的**信任鸿沟**是感知准确性：Lean 证明的策略安全性和活性是在"感知输出完全正确"的前提下成立的。Python 端通过鲁棒性测评（颜色变体 + 空间变体测试）来建立对感知模块实际准确性的信心。