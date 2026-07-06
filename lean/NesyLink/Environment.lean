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

-- 出口类型
inductive ExitType where
  | normal
  | lockedKey
  | conditional
  deriving DecidableEq, Repr

structure SymbolicState where
  player      : Position
  walls       : List Position
  health      : Nat
  keys        : Nat
  monsters    : List Position
  chests      : List Position
  traps       : List Position
  exits       : List Position
  gold        : Nat
  buttons     : List Position
  roomId      : String
  deriving DecidableEq, Repr

def inBounds (p : Position) : Prop :=
  p.1 < 10 ∧ p.2 < 8

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop :=
  manhattan a b = 1

def isWalkable (s : SymbolicState) (p : Position) : Prop :=
  inBounds p ∧ p ∉ s.walls

def isSafe (s : SymbolicState) (p : Position) : Prop :=
  isWalkable s p ∧ p ∉ s.traps ∧ p ∉ s.monsters

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
