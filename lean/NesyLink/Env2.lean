-- 状态转移形式化
import NesyLink.Env1

-----1.辅助函数-----

-- 将自然数转化成Fin 8
def toFin8 (n : Nat) : Fin 8 :=
  if h : n < 8 then ⟨n, h⟩
  else ⟨7, by decide⟩

-- 将自然数转化成Fin 10
def toFin10 (n : Nat) : Fin 10 :=
  if h : n < 10 then ⟨n, h⟩
  else ⟨9, by decide⟩

-- 查找列表第n个元素
def nth {α : Type} (l : List α) (n : Nat) : Option α :=
  match l with
  | [] => none
  | x :: xs =>
    if n = 0 then some x
    else nth xs (n - 1)

-----辅助函数完-----

-----2.获取游戏数据-----

-- 得到当前位置指定方向上的下一个位置
def front (coord : Coord) (d : Direction) : Coord :=
  match d with
  | Direction.up => (toFin8 (coord.fst.val - 1), toFin10 (coord.snd.val))
  | Direction.down => (toFin8 (coord.fst.val + 1), toFin10 (coord.snd.val))
  | Direction.left => (toFin8 (coord.fst.val), toFin10 (coord.snd.val - 1))
  | Direction.right => (toFin8 (coord.fst.val), toFin10 (coord.snd.val + 1))

-- 得到当前游戏状态下玩家所处房间
def getRoom (s : GameState) : Room :=
  s.rooms s.player.room

-- 得到指定房间指定位置的地块
def getTile (room : Room) (coord : Coord) : Tile :=
  match nth room.layout coord.fst.val with
  | none => Tile.ground
  | some row =>
    match nth row coord.snd.val with
    | none => Tile.ground
    | some tile => tile

-- 判断当前地块是否是宝箱
def isChest (t : Tile) : Bool :=
  match t with
  | Tile.chest _ _ _ _ => true
  | _ => false

-- 判断当前地块是否是开关
def isSwitch (t : Tile) : Bool :=
  match t with
  | Tile.switch _ => true
  | _ => false

-- 查找指定房间指定位置的怪物
def findEnemy (room : Room) (coord : Coord) : Option Enemy :=
  room.enemies.find? (fun e => e.coord.fst.val == coord.fst.val ∧ e.coord.snd.val == coord.snd.val)

-- 查找指定房间指定位置的宝箱
def findChest (room : Room) (coord : Coord) : Option Tile :=
  let tile := getTile room coord
  if isChest tile then some tile else none

-- 查找指定房间指定位置的开关
def findSwitch (room : Room) (coord : Coord) : Option Tile :=
  let tile := getTile room coord
  if isSwitch tile then some tile else none

-- 查找指定房间指定编号的门
def findDoor (room : Room) (idx : Nat) : Option DoorInfo :=
  room.doors.find? (fun d => d.id = idx)

-- 根据玩家位置和怪物位置判断怪物远离玩家的方向
def moveAway (player : Coord) (enemy : Coord) : Direction :=
  let dr := (player.fst.val : Int) - (enemy.fst.val : Int)
  let dc := (player.snd.val : Int) - (enemy.snd.val : Int)
  if dr.natAbs >= dc.natAbs then
    if dr > 0 then Direction.up else Direction.down
  else
    if dc > 0 then Direction.left else Direction.right

-- 根据玩家位置和怪物位置判断怪物靠近玩家的方向
def moveTowardsPlayer (player : Coord) (enemy : Coord) : Direction :=
  let (px, py) := (player.fst.val, player.snd.val)
  let (ex, ey) := (enemy.fst.val, enemy.snd.val)
  if px < ex then
    Direction.up
  else if px > ex then
    Direction.down
  else if py < ey then
    Direction.left
  else
    Direction.right

-- 判断玩家和怪物是否相邻
def near (player : Coord) (enemy : Coord) : Bool :=
  let dr := (player.fst.val : Int) - (enemy.fst.val : Int)
  let dc := (player.snd.val : Int) - (enemy.snd.val : Int)
  dr.natAbs + dc.natAbs <= 1

def isEnemyNear (enemy : Enemy) (player : Coord) : Bool :=
  near player enemy.coord

-- 检查当前游戏状态下条件是否满足
def conditionSatisfied (s : GameState) (cond : Condition) : Bool :=
  match cond with
  | Condition.None => true
  | Condition.consumeKey => s.player.key > 0
  | Condition.ButtonPressed buttonPos =>
    let tile := getTile (getRoom s) buttonPos
    match tile with
    | Tile.button pressed => pressed
    | _ => false
  | Condition.EnimyCleared =>
    (getRoom s).enemies.isEmpty

-- 判断玩家朝向和门的朝向是否符合
def orientationFit (d : Direction) (doorinfo : DoorInfo) : Bool :=
  match d, doorinfo.orientation with
  | Direction.up, Direction.up => true
  | Direction.down, Direction.down => true
  | Direction.left, Direction.left => true
  | Direction.right, Direction.right => true
  | _, _ => false

-----获取游戏数据完-----

-----3.更新游戏数据-----

-- 以指定规则更新房间布局
def updateLayoutAt (layout : List (List Tile)) (r c : Nat) (f : Tile → Tile) : List (List Tile) :=
  match layout, r with
  | [], _ => []
  | row :: rows, 0 => (row.mapIdx (fun j tile => if j = c then f tile else tile)) :: rows
  | row :: rows, n+1 => row :: updateLayoutAt rows n c f

-- 证明更新后房间长度满足要求
theorem updateLayoutAt_length (layout : List (List Tile)) (r c : Nat) (f : Tile → Tile) :
    (updateLayoutAt layout r c f).length = layout.length := by
  induction layout generalizing r with
  | nil => rfl
  | cons row rows ih =>
    cases r with
    | zero => rfl
    | succ r' => simp [updateLayoutAt, ih]

-- 证明更新后房间宽度满足要求
theorem updateLayoutAt_width (layout : List (List Tile)) (r c : Nat) (f : Tile → Tile)
    (h : ∀ row ∈ layout, row.length = 10) :
    ∀ row' ∈ updateLayoutAt layout r c f, row'.length = 10 := by
  induction layout generalizing r with
  | nil => simp [updateLayoutAt]
  | cons row rows ih =>
    cases r with
    | zero =>
      simp [updateLayoutAt]
      constructor
      · exact h row (by simp)
      · intro row' hrow'
        exact h row' (by simp [hrow'])
    | succ r' =>
      simp [updateLayoutAt]
      constructor
      · exact h row (by simp)
      · have h_rows : ∀ row' ∈ rows, row'.length = 10 := by
          intro r'' hr''; exact h r'' (by simp [hr''])
        exact ih r' h_rows

-- 将怪物向指定方向移动
def moveEnemy (room : Room) (enemy : Coord) (d : Direction) : Coord :=
  let nextTile := getTile room (front enemy d)
  match nextTile with
  | Tile.ground => front enemy d
  | Tile.wall => enemy
  | Tile.spike => front enemy d
  | Tile.button _ => front enemy d
  | Tile.switch _ => front enemy d
  | Tile.chest _ _ _ _ => enemy
  | Tile.bridge _ _ _ => front enemy d
  | Tile.door _ => front enemy d

-- 玩家在一个状态下攻击面向的方向的怪物，返回攻击后游戏状态
def killEnemy (s : GameState) (coord : Coord) : GameState :=
  let enemy := findEnemy (getRoom s) coord
  match enemy with
  | none => s
  | some e =>
    if s.player.hasSword then
      if e.hp <= 1 then
        let newEnemies := (getRoom s).enemies.filter (fun en => en.coord != coord)
        let newRoom := { (getRoom s) with enemies := newEnemies}
        { s with rooms := fun r => if r = s.player.room then newRoom else s.rooms r, player := {s.player with gold := s.player.gold + 2}, round := s.round + 1 }
      else
        let newEnemies := (getRoom s).enemies.map (fun en => if en.coord = coord then { en with hp := en.hp - 1, coord := moveEnemy (getRoom s) en.coord (moveAway s.player.coord en.coord) } else en)
        let newRoom := { (getRoom s) with enemies := newEnemies}
        { s with rooms := fun r => if r = s.player.room then newRoom else s.rooms r, round := s.round + 1 }
    else
      { s with round := s.round + 1 }

-- 玩家在一个状态下打开指定位置的宝箱，返回打开后游戏状态
def openChest (s : GameState) (coord : Coord) : GameState :=
  let room := getRoom s
  let chest := findChest room coord
  match chest with
  | some (Tile.chest opened content hidden cond) =>
    if opened then { s with round := s.round + 1 }
    else
      if conditionSatisfied s cond then
        let newlayout := updateLayoutAt room.layout coord.fst.val coord.snd.val (fun _ => Tile.chest true content hidden cond)
        have h_height : newlayout.length = 8 := by
          rw [updateLayoutAt_length, room.inv_height]
        have h_width : ∀ row ∈ newlayout, row.length = 10 :=
          updateLayoutAt_width room.layout coord.fst.val coord.snd.val (fun _ => Tile.chest true content hidden cond) room.inv_width
        let newRoom := { room with layout := newlayout, inv_height := h_height, inv_width := h_width }
        if content == Item.key then
          { s with rooms := fun r => if r = s.player.room then newRoom else s.rooms r, player := { s.player with key := s.player.key + 1 }, round := s.round + 1 }
        else
          { s with rooms := fun r => if r = s.player.room then newRoom else s.rooms r, player := { s.player with hasSword := true }, round := s.round + 1 }
      else { s with round := s.round + 1 }
  | _ => { s with round := s.round + 1 }

-- 玩家在指定房间指定位置切换开关，返回切换后的房间信息
def toggleSwitch (room : Room) (coord : Coord) : Room :=
  let switch := findSwitch room coord
  match switch with
  | some (Tile.switch state) =>
    let newlayout := updateLayoutAt room.layout coord.fst.val coord.snd.val (fun _ => Tile.switch ((state + 1) % 3))
    have h_height : newlayout.length = 8 := by
      rw [updateLayoutAt_length, room.inv_height]
    have h_width : ∀ row ∈ newlayout, row.length = 10 :=
      updateLayoutAt_width room.layout coord.fst.val coord.snd.val (fun _ => Tile.switch ((state + 1) % 3)) room.inv_width
    { room with layout := newlayout, inv_height := h_height, inv_width := h_width }
  | _ => room

-- 玩家通过了指定的门之后玩家的信息
def updatePlayer (doorinfo : DoorInfo) (player : Player) : Player :=
  if doorinfo.isOpened then
    { player with room := doorinfo.targetRoom, coord := doorinfo.targetCoord }
  else
    if doorinfo.condition == Condition.consumeKey then
      { player with room := doorinfo.targetRoom, coord := doorinfo.targetCoord, key := player.key - 1 }
    else
      { player with room := doorinfo.targetRoom, coord := doorinfo.targetCoord }

-- 玩家在指定房间指定位置按下按钮，返回按下后的房间信息
def pushButton (room : Room) (coord : Coord) : Room :=
  let newlayout := updateLayoutAt room.layout coord.fst.val coord.snd.val (fun _ => Tile.button true)
  have h_height : newlayout.length = 8 := by
    rw [updateLayoutAt_length, room.inv_height]
  have h_width : ∀ row ∈ newlayout, row.length = 10 :=
    updateLayoutAt_width room.layout coord.fst.val coord.snd.val (fun _ => Tile.button true) room.inv_width
  { room with layout := newlayout, inv_height := h_height, inv_width := h_width }

-- 玩家所在房间怪物向玩家移动一步之后的房间信息
def updateEnemy (room : Room) (player : Coord) : Room :=
  let newEnemies := room.enemies.map (fun e =>
      let d := moveTowardsPlayer player e.coord
      { e with coord := moveEnemy room e.coord d })
  { room with enemies := newEnemies}

-- 根据玩家是否开盾以及房间信息和玩家位置判断玩家扣血情况
def updateHealth (input : Input) (room : Room) (player : Coord) (health : Nat) : Nat :=
  let enemies := room.enemies.filter (fun e => isEnemyNear e player)
  if enemies.isEmpty then health
  else
    let newHealth := health - enemies.length
    if input.action == Action.buttonB then
      health
    else
      newHealth

-- 根据房间信息和玩家位置，将和玩家相邻的怪物反弹
def bound (room : Room) (player : Coord) : Room :=
  let newEnemies := room.enemies.map (fun e =>
      if isEnemyNear e player then
        let d := moveAway player e.coord
        { e with coord := moveEnemy room e.coord d }
      else
        e)
  { room with enemies := newEnemies }

-----更新游戏数据完-----

-----4.处理事件-----

-- 处理等待
def handleWait (input : Input) (s :GameState) : GameState :=
  let s1 :=
    if s.round % 2 = 0 then
      let newHealth := updateHealth input (getRoom s) s.player.coord s.player.health
      let newRoom := bound (getRoom s) s.player.coord
      let finalRoom := updateEnemy newRoom s.player.coord
      { s with player := { s.player with health := newHealth },
               rooms := fun r => if r = s.player.room then finalRoom else s.rooms r }
    else
      s
  { s1 with round := s1.round + 1}


-- 处理移动
def handleMove (input : Input) (s :GameState) (d : Direction) : GameState :=
  let s1 :=
    if s.round % 2 = 0 then
      let newHealth := updateHealth input (getRoom s) s.player.coord s.player.health
      let newRoom := bound (getRoom s) s.player.coord
      let finalRoom := updateEnemy newRoom s.player.coord
      { s with player := { s.player with health := newHealth },
               rooms := fun r => if r = s.player.room then finalRoom else s.rooms r }
    else
      s
  let nextTile := getTile (getRoom s1) (front s1.player.coord d)
  match nextTile with
  | Tile.ground =>
    { s1 with player := { s1.player with coord := front s1.player.coord d }, round := s1.round + 1 }
  | Tile.wall => { s1 with round := s1.round + 1 }
  | Tile.spike =>
    { s1 with player := { s1.player with health := s1.player.health - 1, coord := (s1.rooms (s1.player.room)).spawn }, round := s1.round + 1 }
  | Tile.button _ =>
    let newRoom := pushButton (getRoom s1) (front s1.player.coord d)
    { s1 with rooms := fun r => if r = s1.player.room then newRoom else s1.rooms r, round := s1.round + 1 }
  | Tile.switch _ =>
    { s1 with player := { s1.player with coord := front s1.player.coord d }, round := s1.round + 1 }
  | Tile.chest _ _ _ _ => { s1 with round := s1.round + 1 }
  | Tile.bridge switchRoom switchCoord activeState =>
    if getTile (s1.rooms switchRoom) switchCoord == Tile.switch activeState then
      { s1 with player := { s1.player with coord := front s1.player.coord d }, round := s1.round + 1 }
    else
      { s1 with round := s1.round + 1 }
  | Tile.door idx =>
    match findDoor (getRoom s1) idx with
    | none =>  { s1 with player := { s1.player with coord := front s1.player.coord d }, round := s1.round + 1 }
    | some doorinfo =>
      if orientationFit d doorinfo ∧ conditionSatisfied s1 doorinfo.condition then
        let newPlayer := updatePlayer doorinfo s1.player
        { s1 with player := newPlayer, round := s1.round + 1 }
      else
        { s1 with player := { s1.player with coord := front s1.player.coord d }, round := s1.round + 1 }

-- 处理交互
def handleInteract (input : Input) (s : GameState) (d : Direction) : GameState :=
  let nextTile := getTile (getRoom s) (front s.player.coord d)
  let s0 :=
     match findEnemy (getRoom s) (front s.player.coord d) with
    | some _ =>
      killEnemy s (front s.player.coord d)
    | none =>
      match nextTile with
      | Tile.ground => { s with round := s.round + 1 }
      | Tile.wall => { s with round := s.round + 1 }
      | Tile.spike => { s with round := s.round + 1 }
      | Tile.button _ => { s with round := s.round + 1 }
      | Tile.switch _ =>
        { s with rooms := fun r => if r = s.player.room then toggleSwitch (getRoom s) (front s.player.coord d) else s.rooms r, round := s.round + 1 }
      | Tile.chest _ _ _ _ =>
        openChest s (front s.player.coord d)
      | Tile.bridge _ _ _ => { s with round := s.round + 1 }
      | Tile.door _ => { s with round := s.round + 1 }
  if s.round % 2 = 0 then
    let newHealth := updateHealth input (getRoom s0) s0.player.coord s0.player.health
    let newRoom := bound (getRoom s0) s0.player.coord
    let finalRoom := updateEnemy newRoom s0.player.coord
    { s0 with player := { s0.player with health := newHealth },
              rooms := fun r => if r = s0.player.room then finalRoom else s0.rooms r }
  else
    s0

-- 处理开盾
def handleDefense (input : Input) (s :GameState) : GameState :=
  let s1 :=
    if s.round % 2 = 0 then
      let newHealth := updateHealth input (getRoom s) s.player.coord s.player.health
      let newRoom := bound (getRoom s) s.player.coord
      let finalRoom := updateEnemy newRoom s.player.coord
      { s with player := { s.player with health := newHealth },
               rooms := fun r => if r = s.player.room then finalRoom else s.rooms r }
    else
      s
  { s1 with round := s1.round + 1}

-----处理事件完-----

-----5.状态转移-----

-- 根据现有状态和输入进行状态转移
inductive step : GameState -> Input → GameState -> Prop where
  -- 等待
  | wait (s : GameState) (d : Direction) :
      step s { direction := d, action := Action.wait } (handleWait { direction := d, action := Action.wait } s)
  -- 移动
  | move (s : GameState) (d : Direction):
     step s { direction := d, action := Action.move } (handleMove { direction := d, action := Action.move } s d)
  -- 交互
  | interact (s : GameState) (d : Direction) :
      step s { direction := d, action := Action.buttonA } (handleInteract { direction := d, action := Action.buttonA } s d)
  -- 开盾防御，在怪物行动中进行判定，这里等价于wait
  | defense (s : GameState) (d : Direction) :
      step s { direction := d, action := Action.buttonB } (handleDefense { direction := d, action := Action.buttonB } s)

-----状态转移完-----
