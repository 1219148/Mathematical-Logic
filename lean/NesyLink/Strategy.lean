import NesyLink.Env1

-- ==========================================
-- 1. 扩充的抽象接口 (包装底层复杂的集合与特征判定)
-- ==========================================

-- 地图实体检索
opaque getCurrentMonsters (gs : GameState) : List Coord
opaque getCurrentChests (gs : GameState) : List Coord
opaque getCurrentSwitches (gs : GameState) : List Coord
opaque getCurrentExits (gs : GameState) : List Coord

-- 空间拓扑判定
opaque isAdjacent (c1 c2 : Coord) : Bool
opaque getDirectionTo (fromCoord toCoord : Coord) : Option Direction
opaque nextStepTo (start : Coord) (goals : List Coord) (blocked : List Coord) : Option Direction

-- 关卡结构特征判定
opaque hasBridge (gs : GameState) : Bool
opaque getBridgeSides (gs : GameState) : List Direction
opaque isKnownSwitchRoom (gs : GameState) : Bool
opaque bridgePrimarySide (gs : GameState) : Option Direction
opaque centerExitSide (gs : GameState) : Option Direction

-- 决策辅助抽象
opaque chooseVisibleExitSide (gs : GameState) : Option Direction
opaque sideApproachGoals (gs : GameState) (dir : Direction) (blocked : List Coord) : List Coord
opaque chooseTargetSide (gs : GameState) (as : AgentState) : Option Direction
opaque isTargetSideReachable (gs : GameState) (target : Direction) : Bool


-- ==========================================
-- 2. 状态与目标定义 (保持不变)
-- ==========================================

structure AgentState where
  hubExplorationStarted   : Bool
  rememberedBlockers      : List Coord
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

inductive ObjectiveKind where
  | fight
  | interact
  | navigate
  | goExit
  | idle
  deriving DecidableEq, Repr

inductive InteractionKind where
  | chest
  | switch
  | button
  | monster
  deriving DecidableEq, Repr

structure Objective where
  kind : ObjectiveKind
  targets : List Coord
  side : Option Direction
  interactionKind : Option InteractionKind


-- ==========================================
-- 3. 补全：中心室与拉杆房复杂逻辑 (_choose_bridge_objective)
-- ==========================================

def chooseBridgeObjective (gs : GameState) (as : AgentState) : Objective × AgentState :=
  let isHub := hasBridge gs || (getCurrentExits gs).length > 1

  -- 对应 Python 行为：如果回到了 Hub 且踩下了开关，重置旋转桥状态锁
  let as1 := if isHub && as.postGoalSwitchPressed then
               { as with postGoalRotateBridge := false, postGoalSwitchPressed := false }
             else as

  if isHub then
    -- 分支 A: 怪物清完了，有宝箱，去开宝箱
    if as1.monsterObjectiveDone && !(getCurrentChests gs).isEmpty then
      ({ kind := ObjectiveKind.interact, targets := getCurrentChests gs, side := Option.none, interactionKind := Option.some InteractionKind.chest }, as1)

    -- 分支 B: 怪物清完了，需要旋转桥，且准备去拉杆房
    else if as1.monsterObjectiveDone && as1.postGoalRotateBridge then
      match as1.switchHubSide with
      | Option.some side => ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some side, interactionKind := Option.none }, as1)
      | Option.none      => ({ kind := ObjectiveKind.idle, targets := [], side := Option.none, interactionKind := Option.none }, as1)

    -- 分支 C: 怪物清完了的通用导航（边界扫描）
    else if as1.monsterObjectiveDone then
      ({ kind := ObjectiveKind.navigate, targets := getCurrentChests gs, side := Option.none, interactionKind := Option.none }, as1)

    -- 分支 D: 基础 Hub 房间导航（寻找目标出口或前往拉杆房激活桥）
    else
      match chooseTargetSide gs as1 with
      | Option.none => ({ kind := ObjectiveKind.idle, targets := [], side := Option.none, interactionKind := Option.none }, as1)
      | Option.some targetSide =>
        if isTargetSideReachable gs targetSide then
          let as2 := { as1 with pressedSwitchForTarget := false }
          ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some targetSide, interactionKind := Option.none }, as2)
        else
          let switchSide := match as1.switchHubSide with | Option.some s => Option.some s | Option.none => bridgePrimarySide gs
          let as2 := { as1 with pressedSwitchForTarget := false }
          match switchSide with
          | Option.some s => ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some s, interactionKind := Option.none }, as2)
          | Option.none   => ({ kind := ObjectiveKind.idle, targets := [], side := Option.none, interactionKind := Option.none }, as2)
  else
    -- 处于拉杆房 (Switch Room) 的逻辑
    if isKnownSwitchRoom gs && as1.monsterObjectiveDone && as1.postGoalRotateBridge then
      if !as1.postGoalSwitchPressed then
        ({ kind := ObjectiveKind.interact, targets := getCurrentSwitches gs, side := Option.none, interactionKind := Option.some InteractionKind.switch }, as1)
      else
        match centerExitSide gs with
        | Option.some side => ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some side, interactionKind := Option.none }, as1)
        | Option.none => ({ kind := ObjectiveKind.navigate, targets := getCurrentExits gs, side := Option.none, interactionKind := Option.none }, as1)

    else if isKnownSwitchRoom gs && as1.currentTargetSide.isSome && !as1.pressedSwitchForTarget then
      ({ kind := ObjectiveKind.interact, targets := getCurrentSwitches gs, side := Option.none, interactionKind := Option.some InteractionKind.switch }, as1)

    else
      -- 离开拉杆房，并将当前方向标记为已探索
      let as2 := match centerExitSide gs with
                 | Option.some side => { as1 with exploredHubSides := side :: as1.exploredHubSides }
                 | Option.none      => as1
      match centerExitSide gs with
      | Option.some side => ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some side, interactionKind := Option.none }, as2)
      | Option.none => ({ kind := ObjectiveKind.navigate, targets := getCurrentExits gs, side := Option.none, interactionKind := Option.none }, as2)


-- ==========================================
-- 4. 顶层决策桥接 (_choose_local_objective)
-- ==========================================

def chooseLocalObjective (gs : GameState) (as : AgentState) (useHubExploration : Bool) : Objective × AgentState :=
  let monsters := getCurrentMonsters gs
  let hasSword := gs.player.hasSword

  -- 1. 战斗优先
  if !monsters.isEmpty && (hasSword || !useHubExploration) then
    ({ kind := ObjectiveKind.fight, targets := monsters, side := Option.none, interactionKind := Option.some InteractionKind.monster }, as)

  else
    let chests := getCurrentChests gs
    let validChests := chests.filter (fun c => !as.rememberedBlockers.contains c)

    -- 2. 核心开宝箱
    if !validChests.isEmpty && (gs.player.key == 0 || useHubExploration) then
      ({ kind := ObjectiveKind.interact, targets := validChests, side := Option.none, interactionKind := Option.some InteractionKind.chest }, as)

    -- 3. 补全：调用中心室/拉杆房特定策略
    else if useHubExploration then
      chooseBridgeObjective gs as

    -- 4. 普通寻找出口
    else
      match chooseVisibleExitSide gs with
      | Option.some side => ({ kind := ObjectiveKind.goExit, targets := [], side := Option.some side, interactionKind := Option.none }, as)
      | Option.none => ({ kind := ObjectiveKind.navigate, targets := getCurrentExits gs, side := Option.none, interactionKind := Option.none }, as)


-- ==========================================
-- 5. 动作执行器 (保持修正后的 Match-With 语法)
-- ==========================================

def executeObjective (gs : GameState) (as : AgentState) (obj : Objective) : Input × AgentState :=
  let pCoord := gs.player.coord
  let blockers := as.rememberedBlockers

  match obj.kind with
  | ObjectiveKind.idle =>
    ({ direction := Direction.up, action := Action.wait }, as)

  | ObjectiveKind.fight =>
    let adjacentMonsters := obj.targets.filter (fun m => isAdjacent pCoord m)
    match adjacentMonsters.head? with
    | Option.some m =>
      match getDirectionTo pCoord m with
      | Option.some dir => ({ direction := dir, action := Action.buttonA }, as)
      | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)
    | Option.none =>
      match nextStepTo pCoord obj.targets blockers with
      | Option.some dir => ({ direction := dir, action := Action.move }, as)
      | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)

  | ObjectiveKind.interact =>
    let adjacentTargets := obj.targets.filter (fun t => isAdjacent pCoord t)
    match adjacentTargets.head? with
    | Option.some t =>
      match getDirectionTo pCoord t with
      | Option.some dir => ({ direction := dir, action := Action.buttonA }, as)
      | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)
    | Option.none =>
      match nextStepTo pCoord obj.targets blockers with
      | Option.some dir => ({ direction := dir, action := Action.move }, as)
      | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)

  | ObjectiveKind.goExit =>
    match obj.side with
    | Option.some side =>
      let approach := sideApproachGoals gs side blockers
      if approach.contains pCoord then
        ({ direction := side, action := Action.move }, as)
      else
        match nextStepTo pCoord approach blockers with
        | Option.some dir => ({ direction := dir, action := Action.move }, as)
        | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)
    | Option.none => ({ direction := Direction.up, action := Action.wait }, as)

  | ObjectiveKind.navigate =>
    match nextStepTo pCoord obj.targets blockers with
    | Option.some dir => ({ direction := dir, action := Action.move }, as)
    | Option.none     => ({ direction := Direction.up, action := Action.wait }, as)


-- ==========================================
-- 6. 补全：顶层局部策略接口 (actLocalPlanner)
-- ==========================================

def actLocalPlanner (gs : GameState) (as : AgentState) : Input × AgentState :=
  -- 补全：根据 Python 语义判定当前房间是否具有中心室（Hub）特征
  let isHubRoom : Bool := hasBridge gs || (getCurrentExits gs).length > 1
  let useHubExploration := isHubRoom || as.hubExplorationStarted

  let as' := { as with hubExplorationStarted := useHubExploration }

  -- 接收选择的目标以及同步更新后的智能体状态
  let (obj, as'') := chooseLocalObjective gs as' useHubExploration

  executeObjective gs as'' obj



-- SECTION

-- ==========================================
-- 1. 全局规划器所需的抽象大地图接口
-- ==========================================

structure RoomId where
  id : Nat
  deriving DecidableEq, Repr, Inhabited, Nonempty  -- 👈 修复1：增加了 Inhabited 和 Nonempty

-- 全局拓扑与路由黑盒抽象
opaque getCurrentRoomId (gs : GameState) : RoomId
opaque isDungeonExit (room : RoomId) : Bool
opaque getDungeonExitDirection (gs : GameState) : Option Direction
opaque getNextTargetRoom (gs : GameState) (gas : GlobalAgentState) : Option RoomId
opaque getNextRoomOnPath (start : RoomId) (target : RoomId) : Option RoomId
opaque getDirectionToNeighborRoom (current : RoomId) (next : RoomId) : Option Direction
opaque needsKeyToUnlock (room : RoomId) : Bool


-- ==========================================
-- 2. 全局智能体状态与宏观目标定义
-- ==========================================

inductive GlobalObjective where
  | clearCurrentRoom
  | goToRoom (target : RoomId)
  | unlockDoorAt (target : RoomId)
  | escapeDungeon
  deriving DecidableEq, Repr

structure GlobalAgentState where
  currentRoom       : RoomId
  visitedRooms      : List RoomId
  unlockedDoors     : List RoomId
  macroObjective    : GlobalObjective
  localAgentState   : AgentState


-- ==========================================
-- 3. 辅助函数：局部规划器内存状态重置器
-- ==========================================

def initialLocalAgentState : AgentState := {
  hubExplorationStarted   := false,
  rememberedBlockers      := [],
  monsterObjectiveDone    := false,
  sawMonsterObjective     := false,
  monsterAbsenceTicks     := 0,
  currentTargetSide       := Option.none,
  switchHubSide           := Option.none,
  hubSwitchPositions      := [],
  pressedSwitchForTarget  := false,
  exploredHubSides        := [],
  postGoalRotateBridge    := false,
  postGoalSwitchPressed   := false
}


-- ==========================================
-- 4. 跨房间转移管理 (Room Transition Handler)
-- ==========================================

def handleRoomTransition (gs : GameState) (gas : GlobalAgentState) : GlobalAgentState :=
  let actualRoom := getCurrentRoomId gs
  if actualRoom != gas.currentRoom then  -- 👈 修复2：将 /= 修改为了 !=
    let updatedVisited := if gas.visitedRooms.contains actualRoom then gas.visitedRooms else actualRoom :: gas.visitedRooms
    let updatedUnlocked := if needsKeyToUnlock actualRoom && !(gas.unlockedDoors.contains actualRoom)
                           then actualRoom :: gas.unlockedDoors
                           else gas.unlockedDoors
    { gas with
      currentRoom     := actualRoom,
      visitedRooms    := updatedVisited,
      unlockedDoors   := updatedUnlocked,
      localAgentState := initialLocalAgentState
    }
  else
    gas


-- ==========================================
-- 5. 全局宏观目标决策逻辑 (Global Objective Selector)
-- ==========================================

def updateGlobalObjective (gs : GameState) (gas : GlobalAgentState) : GlobalObjective :=
  let curRoom := getCurrentRoomId gs

  if isDungeonExit curRoom && (getCurrentMonsters gs).isEmpty then
    GlobalObjective.escapeDungeon
  else
    let monsters := getCurrentMonsters gs
    let chests := getCurrentChests gs
    let unblockedChests := chests.filter (fun c => !gas.localAgentState.rememberedBlockers.contains c)

    if !monsters.isEmpty || !unblockedChests.isEmpty then
      GlobalObjective.clearCurrentRoom
    else
      match getNextTargetRoom gs gas with
      | Option.none => GlobalObjective.escapeDungeon
      | Option.some targetRoom =>
        if needsKeyToUnlock targetRoom && gs.player.key > 0 then
          GlobalObjective.unlockDoorAt targetRoom
        else
          GlobalObjective.goToRoom targetRoom


-- ==========================================
-- 6. 全局宏观目标执行与控制下发 (Global Execution)
-- ==========================================

def executeGlobalObjective (gs : GameState) (gas : GlobalAgentState) (macroGoal : GlobalObjective) : Input × GlobalAgentState :=
  match macroGoal with
  | GlobalObjective.clearCurrentRoom =>
    let (input, nextLocalState) := actLocalPlanner gs gas.localAgentState
    (input, { gas with localAgentState := nextLocalState })

  | GlobalObjective.goToRoom targetRoom =>
    match getNextRoomOnPath gas.currentRoom targetRoom with
    | Option.none => ({ direction := Direction.up, action := Action.wait }, gas)
    | Option.some nextRoom =>
      match getDirectionToNeighborRoom gas.currentRoom nextRoom with
      | Option.none => ({ direction := Direction.up, action := Action.wait }, gas)
      | Option.some exitDir =>
        let obj := { kind := ObjectiveKind.goExit, targets := [], side := Option.some exitDir, interactionKind := Option.none }
        let (input, nextLocalState) := executeObjective gs gas.localAgentState obj
        (input, { gas with localAgentState := nextLocalState })

  | GlobalObjective.unlockDoorAt targetRoom =>
    match getNextRoomOnPath gas.currentRoom targetRoom with
    | Option.none => ({ direction := Direction.up, action := Action.wait }, gas)
    | Option.some nextRoom =>
      match getDirectionToNeighborRoom gas.currentRoom nextRoom with
      | Option.none => ({ direction := Direction.up, action := Action.wait }, gas)
      | Option.some exitDir =>
        let obj := { kind := ObjectiveKind.goExit, targets := [], side := Option.some exitDir, interactionKind := Option.none }
        let (input, nextLocalState) := executeObjective gs gas.localAgentState obj
        (input, { gas with localAgentState := nextLocalState })

  | GlobalObjective.escapeDungeon =>
    match getDungeonExitDirection gs with
    | Option.some exitDir =>
      let obj := { kind := ObjectiveKind.goExit, targets := [], side := Option.some exitDir, interactionKind := Option.none }
      let (input, nextLocalState) := executeObjective gs gas.localAgentState obj
      (input, { gas with localAgentState := nextLocalState })
    | Option.none =>
      let (input, nextLocalState) := actLocalPlanner gs gas.localAgentState
      (input, { gas with localAgentState := nextLocalState })


-- ==========================================
-- 7. 顶层端到端全局规划器接口 (Top-level Entry)
-- ==========================================

def actGlobalPlanner (gs : GameState) (gas : GlobalAgentState) : Input × GlobalAgentState :=
  let gas1 := handleRoomTransition gs gas
  let nextMacroGoal := updateGlobalObjective gs gas1
  let gas2 := { gas1 with macroObjective := nextMacroGoal }
  executeGlobalObjective gs gas2 nextMacroGoal
