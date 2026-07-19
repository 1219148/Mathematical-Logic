-- 游戏环境形式化
def Coord := Fin 8 × Fin 10
deriving instance DecidableEq for Coord

inductive Direction where
  | up
  | down
  | left
  | right
  deriving DecidableEq, Repr, Inhabited

inductive Item where
  | key
  | sword
  deriving DecidableEq, Repr

--===================
inductive Action where
  | wait
  | move
  | buttonA
  | buttonB
  deriving DecidableEq, Repr

structure Input where
 direction : Direction
 action : Action
 deriving Repr

inductive Condition where
  | None
  | consumeKey
  | ButtonPressed (buttonPos : Coord)
  | EnimyCleared
  deriving DecidableEq

inductive Tile where
  | ground
  | wall
  | spike
  | button (pressed : Bool)
  | switch (state : Nat)
  | chest (opened : Bool) (content : Item) (hidden : Bool) (cond : Condition)
  | bridge (switchRoom : Nat) (switchCoord : Coord) (activeState : Nat)
  | door (id : Nat)
  deriving DecidableEq

structure DoorInfo where
  id : Nat
  condition : Condition
  isOpened : Bool
  orientation : Direction
  targetRoom : Nat
  targetCoord : Coord

structure Enemy where
  hp : Nat
  coord : Coord

structure Room where
  spawn : Coord
  layout : List (List Tile)
  doors : List DoorInfo
  enemies : List Enemy
  inv_height : layout.length = 8
  inv_width  : ∀ row ∈ layout, row.length = 10

structure Player where
  room : Nat
  coord : Coord
  health : Nat
  gold : Nat
  key : Nat
  hasSword : Bool

structure GameState where
  round : Nat
  player : Player
  rooms : Nat → Room
