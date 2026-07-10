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

-- 朝向
inductive Direction where
  | up | down | left | right
  deriving DecidableEq, Repr

-- 怪物类型
inductive MonsterType where
  | chaser
  | ambusher
  | patroller
  deriving DecidableEq, Repr

-- 怪物信息
structure MonsterInfo where
  damage : Nat
  hp : Nat
  pos : Position
  monsterType : MonsterType
  deriving DecidableEq, Repr

-- 陷阱类型
inductive TrapType where
  | spike
  | abyss
  deriving DecidableEq, Repr

structure ChestInfo where
  pos : Position
  opened : Bool
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
  trapType  : TrapType := TrapType.spike
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


-- 房间布局 —— 永远不变的静态地图数据
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

-- 游戏状态 —— 随每一步变化的动态数据
structure SymbolicState where
  player       : Position
  facing       : Direction
  health       : Nat
  keys         : Nat
  gold         : Nat
  items        : List String
  monsters     : List MonsterInfo
  monsterTypes : List (Position × MonsterType)
  chests       : List ChestInfo
  hiddenChests : List Position := []
  buttons      : List String := []
  dynamicStates : List (String × String) := []
  exitOpened    : List Position := []
  shieldTicks  : Nat := 0
  roomMap       : List (String × RoomLayout) := []
  room          : RoomLayout
  deriving DecidableEq, Repr

def inBounds (p : Position) : Prop :=
  p.1 < 10 ∧ p.2 < 8

-- 朝向对应的前方格子
def frontOf (p : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.up    => (p.1, if p.2 = 0 then 0 else p.2 - 1)
  | Direction.down  => (p.1, p.2 + 1)
  | Direction.left  => (if p.1 = 0 then 0 else p.1 - 1, p.2)
  | Direction.right => (p.1 + 1, p.2)

-- 从 List MonsterInfo / List ChestInfo 提取所有位置
def monsterPositions (monsters : List MonsterInfo) : List Position :=
  monsters.map (fun m => m.pos)

def chestPositions (chests : List ChestInfo) : List Position :=
  chests.map (fun c => c.pos)

-- 宝箱是否处于打开状态
def isChestOpened (chests : List ChestInfo) (p : Position) : Bool :=
  (chests.find? (fun c => c.pos = p)).any (fun c => c.opened)

-- 更新指定位置的宝箱为已打开
def openChestAt (chests : List ChestInfo) (p : Position) : List ChestInfo :=
  chests.map (fun c => if c.pos = p then { c with opened := true } else c)

-- 出口相关辅助
def dirToAction (dir : ExitDirection) : Action :=
  match dir with
  | ExitDirection.north => Action.up
  | ExitDirection.south => Action.down
  | ExitDirection.west => Action.left
  | ExitDirection.east => Action.right

def markExitOpened (exitOpened : List Position) (p : Position) : List Position :=
  p :: exitOpened

def isExitOpen (exitInfos : List ExitInfo) (exitOpened : List Position) (p : Position) : Bool :=
  match exitInfos.find? (fun e => e.pos = p) with
  | none => false
  | some e => e.exitType = ExitType.normal || p ∈ exitOpened

-- 从 roomMap 查找房间
def lookupRoom (roomMap : List (String × RoomLayout)) (roomId : String) : Option RoomLayout :=
  match roomMap.find? (fun (id, _) => id = roomId) with
  | none => none
  | some (_, layout) => some layout

-- 查找目标房间的出生点坐标
def lookupSpawn (roomMap : List (String × RoomLayout)) (roomId : String) (entryName : String) : Position :=
  match lookupRoom roomMap roomId with
  | none => (0, 0)
  | some layout =>
    match layout.spawns.find? (fun (name, _) => name = entryName) with
    | some (_, pos) => pos
    | none =>
      match layout.spawns.find? (fun (name, _) => name = layout.defaultSpawn) with
      | some (_, pos) => pos
      | none => (0, 0)

-- 从 RoomLayout 初始化房间动态状态
def initChestsFromRoom (layout : RoomLayout) : List ChestInfo :=
  layout.initialChests.map (fun p => { pos := p, opened := false : ChestInfo })

def initMonstersFromRoom (layout : RoomLayout) : List MonsterInfo :=
  layout.initialMonsterTypes.map (fun (pos, mtype) =>
    { pos := pos, hp := 3, damage := 1, monsterType := mtype : MonsterInfo })

def initButtonsFromRoom (layout : RoomLayout) : List String :=
  layout.initialButtons.map (fun (id, _) => id)

-- 构建 roomTransition 后的新状态
def transitionResult (s : SymbolicState) (targetLayout : RoomLayout) (spawnPos : Position) : SymbolicState :=
  { s with
      player := spawnPos
      facing := Direction.down
      monsters := initMonstersFromRoom targetLayout
      monsterTypes := targetLayout.initialMonsterTypes
      chests := initChestsFromRoom targetLayout
      hiddenChests := targetLayout.hiddenChests
      buttons := initButtonsFromRoom targetLayout
      exitOpened := []
      shieldTicks := 0
      room := targetLayout
  }

-- 在当前房间中按名字查找出生点坐标（陷阱重生用）
def lookupRoomSpawn (room : RoomLayout) (entryName : String) : Position :=
  match room.spawns.find? (fun (name, _) => name = entryName) with
  | some (_, pos) => pos
  | none =>
    match room.spawns.find? (fun (name, _) => name = room.defaultSpawn) with
    | some (_, pos) => pos
    | none => (0, 0)

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop :=
  manhattan a b = 1

-- 玩家能交互的距离：同一格或相邻（曼哈顿距离 ≤ 1）
def withinReach (a b : Position) : Prop :=
  manhattan a b ≤ 1

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
  isWalkable s p ∧ (p ∈ bridgeTiles s ∨ p ∉ trapPositions s) ∧ p ∉ monsterPositions s.monsters

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  -- TODO: 实现所有状态转移规则
  | wait (s : SymbolicState) : Step s Action.wait { s with shieldTicks := (if s.shieldTicks > 0 then s.shieldTicks - 1 else 0) }
  | moveUp (s : SymbolicState) : Step s Action.up { s with player := (s.player.1, s.player.2 - 1), facing := Direction.up, shieldTicks := 0 }
  | moveDown (s : SymbolicState) : Step s Action.down { s with player := (s.player.1, s.player.2 + 1), facing := Direction.down, shieldTicks := 0 }
  | moveLeft (s : SymbolicState) : Step s Action.left { s with player := (s.player.1 - 1, s.player.2), facing := Direction.left, shieldTicks := 0 }
  | moveRight (s : SymbolicState) : Step s Action.right { s with player := (s.player.1 + 1, s.player.2), facing := Direction.right, shieldTicks := 0 }
  | attackMonsterHit (s : SymbolicState) (m : MonsterInfo) :
      m ∈ s.monsters →
      m.pos = frontOf s.player s.facing →
      m.hp > 1 →
      Step s Action.buttonA
        { s with
            monsters := s.monsters.map (fun m' => if m'.pos = m.pos then { m' with hp := m'.hp - 1 } else m')
            shieldTicks := 0
        }
  | attackMonsterKill (s : SymbolicState) (m : MonsterInfo) :
      m ∈ s.monsters →
      m.pos = frontOf s.player s.facing →
      m.hp = 1 →
      Step s Action.buttonA
        { s with
            monsters := s.monsters.filter (fun m' => m'.pos ≠ m.pos)
            gold := s.gold + 2
            shieldTicks := 0
        }
  | defense (s : SymbolicState) :
      "shield" ∈ s.items →
      Step s Action.buttonB
        { s with shieldTicks := 6 }
  | openChest (s : SymbolicState) (c : ChestInfo) :
      c ∈ s.chests →
      withinReach c.pos s.player →
      c.opened = false →
      Step s Action.buttonA
        { s with chests := openChestAt s.chests c.pos, gold := s.gold + 2, shieldTicks := 0 }
  -- 怪物接触伤害（无盾）：玩家扣血
  | monsterDamage (s : SymbolicState) (m : MonsterInfo) :
      m ∈ s.monsters →
      withinReach m.pos s.player →
      s.shieldTicks = 0 →
      s.health > 0 →
      Step s Action.wait
        { s with health := s.health - m.damage }
  -- 怪物接触伤害（有盾格挡）：无伤，盾自然衰减 1 步
  | monsterDamageBlocked (s : SymbolicState) (m : MonsterInfo) :
      m ∈ s.monsters →
      withinReach m.pos s.player →
      s.shieldTicks > 0 →
      Step s Action.wait
        { s with shieldTicks := s.shieldTicks - 1 }
  -- 钉刺陷阱：伤害 + 立即传送回出生点
  | stepOnSpike (s : SymbolicState) (t : TrapInfo) :
      t ∈ s.room.traps →
      s.player ∈ (t.pos :: t.area) →
      t.trapType = TrapType.spike →
      s.health > 0 →
      Step s Action.wait
        { s with
            health := s.health - t.damage
            player := lookupRoomSpawn s.room t.respawnTo
            shieldTicks := 0
        }
  -- 深渊陷阱：伤害 + 传送回出生点（Python 有控制锁延迟 + 智能选重生格，此处简化）
  | stepOnAbyss (s : SymbolicState) (t : TrapInfo) :
      t ∈ s.room.traps →
      s.player ∈ (t.pos :: t.area) →
      t.trapType = TrapType.abyss →
      s.health > 0 →
      Step s Action.wait
        { s with
            health := s.health - t.damage
            player := lookupRoomSpawn s.room t.respawnTo
            shieldTicks := 0
        }
  -- 按按钮（用于激活 conditional 出口条件）
  | pressButton (s : SymbolicState) (buttonId : String) (pos : Position) :
      (buttonId, pos) ∈ s.room.initialButtons →
      withinReach pos s.player →
      buttonId ∉ s.buttons →
      Step s Action.buttonA
        { s with buttons := buttonId :: s.buttons, shieldTicks := 0 }
  -- 挥剑落空（前方无怪物且无宝箱/按钮可交互时）
  | swordMiss (s : SymbolicState) :
      (∀ m ∈ s.monsters, m.pos ≠ frontOf s.player s.facing) →
      (∀ c ∈ s.chests, ¬ (withinReach c.pos s.player ∧ c.opened = false)) →
      (∀ p ∈ s.room.initialButtons, ¬ (withinReach p.2 s.player ∧ p.1 ∉ s.buttons)) →
      Step s Action.buttonA
        { s with shieldTicks := 0 }
    -- 走到出口上 → 自动判断类型 → 切换房间
  | roomTransitionNormal (s : SymbolicState) (e : ExitInfo) (targetLayout : RoomLayout) :
      e ∈ s.room.exitInfos →
      e.exitType = ExitType.normal →
      s.player = e.pos →
      lookupRoom s.roomMap e.targetRoomId = some targetLayout →
      Step s (dirToAction e.direction)
        (transitionResult s targetLayout (lookupSpawn s.roomMap e.targetRoomId e.targetEntry))
  | roomTransitionLockedOpen (s : SymbolicState) (e : ExitInfo) (targetLayout : RoomLayout) :
      e ∈ s.room.exitInfos →
      e.exitType = ExitType.lockedKey →
      s.player = e.pos →
      e.pos ∈ s.exitOpened →
      lookupRoom s.roomMap e.targetRoomId = some targetLayout →
      Step s (dirToAction e.direction)
        (transitionResult s targetLayout (lookupSpawn s.roomMap e.targetRoomId e.targetEntry))
  | roomTransitionLockedUnlock (s : SymbolicState) (e : ExitInfo) (targetLayout : RoomLayout) :
      e ∈ s.room.exitInfos →
      e.exitType = ExitType.lockedKey →
      s.player = e.pos →
      (e.pos ∉ s.exitOpened) →
      s.keys ≥ (e.requires.keyCount.getD 1) →
      lookupRoom s.roomMap e.targetRoomId = some targetLayout →
      Step s (dirToAction e.direction)
        (transitionResult
          { s with
              keys := if e.requires.consumeKey then s.keys - (e.requires.keyCount.getD 1) else s.keys
              exitOpened := markExitOpened s.exitOpened e.pos
          }
          targetLayout
          (lookupSpawn s.roomMap e.targetRoomId e.targetEntry))
  | roomTransitionConditional (s : SymbolicState) (e : ExitInfo) (targetLayout : RoomLayout) :
      e ∈ s.room.exitInfos →
      e.exitType = ExitType.conditional →
      s.player = e.pos →
      (e.requires.keyCount.all (fun n => s.keys ≥ n)) →
      (e.requires.buttonId.all (fun id => id ∈ s.buttons)) →
      (e.requires.requiredItem.all (fun item => item ∈ s.items)) →
      (e.requires.allMonstersDefeated → s.monsters = []) →
      lookupRoom s.roomMap e.targetRoomId = some targetLayout →
      Step s (dirToAction e.direction)
        (transitionResult
          { s with
              keys := if e.requires.consumeKey then s.keys - (e.requires.keyCount.getD 1) else s.keys
              exitOpened := markExitOpened s.exitOpened e.pos
          }
          targetLayout
          (lookupSpawn s.roomMap e.targetRoomId e.targetEntry))

-- TODO: theorem moveInBounds : ...
-- TODO: theorem neverEnterWall : ...
-- TODO: theorem attackOnlyRemovesMonsters : ...

end NesyLink
