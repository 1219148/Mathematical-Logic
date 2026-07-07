namespace NesyLink

-- 玩家坐标
abbrev Position := Nat × Nat

-- 动作类型
inductive Action where
  | wait
  | up
  | down
  | left
  | right
  | buttonA    -- 攻击/交互
  | buttonB    -- 举盾
  deriving DecidableEq, Repr

-- 怪物类型
inductive MonsterType where
  | chaser
  | ambusher
  | patroller
  deriving DecidableEq, Repr

-- 陷阱类型
inductive TrapType where
  | normal
  | abyss
  deriving DecidableEq, Repr

-- 出口类型
inductive ExitType where
  | normal
  | lockedKey
  | conditional
  deriving DecidableEq, Repr

-- 出口方向
inductive ExitDirection where
  | north
  | south
  | west
  | east
  deriving DecidableEq, Repr

-- 出口条件
structure ExitRequires where
  keyCount            : Option Nat := none
  consumeKey          : Bool      := false
  buttonId            : Option String := none
  requiredItem        : Option String := none
  allMonstersDefeated : Bool      := false
  deriving DecidableEq, Repr

-- 出口完整信息
structure ExitInfo where
  pos          : Position
  exitType     : ExitType
  direction    : ExitDirection
  targetRoomId : String
  targetEntry  : String
  requires     : ExitRequires
  completeTask : Bool
  deriving DecidableEq, Repr

-- 陷阱完整信息
structure TrapInfo where
  pos       : Position
  damage    : Nat
  respawnTo : String
  trapType  : TrapType := TrapType.normal
  area      : List Position := []
  deriving DecidableEq, Repr

-- 旋转桥的单个状态（stateId → 安全瓦片列表）
structure BridgeState where
  stateId : String
  tiles   : List Position
  deriving DecidableEq, Repr

-- 动态对象（旋转桥等）
structure DynamicObjectInfo where
  objectId     : String
  kind         : String
  initialState : String
  states       : List BridgeState
  deriving DecidableEq, Repr

-- 开关（触发动态对象的状态循环）
structure SwitchInfo where
  objectId   : String
  pos        : Position
  activation : String
  targetId   : String
  effectType : String
  stateOrder : List String
  deriving DecidableEq, Repr

-- ============================================================
-- 房间布局 —— 永远不变的静态地图数据
-- ============================================================
structure RoomLayout where
  walls               : List Position
  spawns              : List (String × Position)
  defaultSpawn        : String
  traps               : List TrapInfo
  exitInfos           : List ExitInfo
  initialChests       : List Position
  hiddenChests        : List Position := []
  initialMonsters     : List Position
  initialMonsterTypes : List (Position × MonsterType)
  initialButtons      : List (String × Position) := []
  npcs                : List Position := []
  switches            : List SwitchInfo := []
  dynamicObjects      : List DynamicObjectInfo := []
  roomId              : String
  deriving DecidableEq, Repr

-- ============================================================
-- 游戏状态 —— 随每一步变化的动态数据
-- ============================================================
structure SymbolicState where
  player       : Position
  health       : Nat
  keys         : Nat
  gold         : Nat
  items        : List String
  monsters     : List Position
  monsterTypes : List (Position × MonsterType)
  chests       : List Position
  hiddenChests : List Position := []
  buttons      : List String := []
  dynamicStates : List (String × String) := []
  room         : RoomLayout
  deriving DecidableEq, Repr

def inBounds (p : Position) : Prop :=
  p.1 < 10 ∧ p.2 < 8

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop :=
  manhattan a b = 1

-- 整个 10×8 房间的所有瓦片坐标
def allRoomTiles : List Position :=
  let rec go (xs : List Nat) : List Position :=
    match xs with
    | [] => []
    | x :: rest => List.map (fun y => (x, y)) (List.range 8) ++ go rest
  go (List.range 10)

-- 陷阱覆盖的所有位置（单点 + 区域陷阱展开）
def trapPositions (s : SymbolicState) : List Position :=
  let rec go (ts : List TrapInfo) : List Position :=
    match ts with
    | [] => []
    | t :: rest => (t.pos :: t.area) ++ go rest
  go s.room.traps

-- 当前激活的桥砖（查 dynamicStates → 对应 BridgeState 的 tiles）
def bridgeTiles (s : SymbolicState) : List Position :=
  match List.find? (fun (e : String × String) => e.1 == "center_bridge") s.dynamicStates with
  | none => []
  | some entry =>
    match List.find? (fun d => d.objectId == "center_bridge") s.room.dynamicObjects with
    | none => []
    | some obj =>
      match List.find? (fun st => st.stateId == entry.2) obj.states with
      | none => []
      | some st => st.tiles

def isWalkable (s : SymbolicState) (p : Position) : Prop :=
  inBounds p ∧ p ∉ s.room.walls

-- 安全条件：可通行 ∧ （在桥上 ∨ 不在陷阱上） ∧ 不在怪物上
def isSafe (s : SymbolicState) (p : Position) : Prop :=
  isWalkable s p ∧ (p ∈ bridgeTiles s ∨ p ∉ trapPositions s) ∧ p ∉ s.monsters

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  -- TODO: 实现所有状态转移规则
  | wait (s : SymbolicState) : Step s Action.wait s
  -- | moveUp ...
  -- | attackMonster ...
  -- | openChest ...
  -- | openDoor ...
  -- | stepOnTrap ...
  -- | pressButton ...
  -- | roomTransition ...
  deriving DecidableEq

-- TODO: theorem moveInBounds : ...
-- TODO: theorem neverEnterWall : ...
-- TODO: theorem attackOnlyRemovesMonsters : ...

end NesyLink
