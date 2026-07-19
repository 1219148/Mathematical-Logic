import NesyLink.Env1
import NesyLink.Env2
import NesyLink.Strategy

-- 安全与活性公理（opaque 函数的行为规约）
-- 注: 所有公理返回类型使用 ¬ 包装以避免 Lean 4.29-rc6 解析器 bug

-- ============================================================
-- 共享辅助定义
-- ============================================================

def coordDist (c1 c2 : Coord) : Nat :=
  ((c1.fst.val : Int) - (c2.fst.val : Int)).natAbs +
  ((c1.snd.val : Int) - (c2.snd.val : Int)).natAbs

def minCoordDist (start : Coord) (goals : List Coord) : Nat :=
  match goals with
  | [] => 0
  | g :: gs => min (coordDist start g) (minCoordDist start gs)

def totalMonsterHP (gs : GameState) : Nat :=
  (getRoom gs).enemies.foldl (fun sum e => sum + e.hp) 0

def blockersCoverObstacles (gs : GameState) (blockers : List Coord) : Prop :=
  ∀ (c : Coord),
    (let t := getTile (getRoom gs) c; t = Tile.wall ∨ t = Tile.spike) →
    c ∈ blockers

-- ============================================================
-- 安全性公理（定理 1 & 2 使用）
-- ============================================================

axiom nextStepTo_avoids_blocked (start : Coord) (goals blocked : List Coord) (d : Direction) (h : nextStepTo start goals blocked = some d) : ¬ (front start d) ∈ blocked

axiom exit_direction_not_wall (gs : GameState) (side : Direction) (blocked : List Coord) (pCoord : Coord) (h : (sideApproachGoals gs side blocked).contains pCoord = true) : ¬ (getTile (getRoom gs) (front pCoord side) = Tile.wall)

axiom exit_direction_not_spike (gs : GameState) (side : Direction) (blocked : List Coord) (pCoord : Coord) (h : (sideApproachGoals gs side blocked).contains pCoord = true) : ¬ (getTile (getRoom gs) (front pCoord side) = Tile.spike)

axiom getDirectionTo_correct (src : Coord) (dst : Coord) (d : Direction) (h : getDirectionTo src dst = some d) : ¬ (¬ (front src d = dst))

axiom isAdjacent_correct (c1 c2 : Coord) : ¬ (¬ (isAdjacent c1 c2 = true ↔ near c1 c2))

-- ============================================================
-- 活性公理（定理 3 & 4 使用）
-- ============================================================

-- nextStepTo 在玩家不邻近任何目标时始终能返回方向（存在路径）
axiom nextStepTo_some (start : Coord) (goals blocked : List Coord) (hGoalsNE : goals ≠ []) (hNotNearAny : ∀ g ∈ goals, near start g = false) : ¬ (¬ (∃ d, nextStepTo start goals blocked = some d))

-- nextStepTo 减小到目标集的最小距离
axiom nextStepTo_progress (start : Coord) (goals blocked : List Coord) (d : Direction) (hNext : nextStepTo start goals blocked = some d) (hNotNearAny : ∀ g ∈ goals, near start g = false) : ¬ (¬ (minCoordDist (front start d) goals < minCoordDist start goals))

-- getDirectionTo 在相邻时必定成功
axiom getDirectionTo_adjacent (c1 c2 : Coord) (hNear : near c1 c2 = true) : ¬ (¬ (∃ d, getDirectionTo c1 c2 = some d))

-- getCurrentMonsters 返回的列表为空 ↔ 房间 enemies 为空
axiom getCurrentMonsters_empty_iff (gs : GameState) : ((getCurrentMonsters gs).isEmpty = false) ↔ (getRoom gs).enemies ≠ []

-- getCurrentMonsters 成员等价于 findEnemy 成功
axiom getCurrentMonsters_mem (gs : GameState) (c : Coord) : (c ∈ getCurrentMonsters gs) ↔ (findEnemy (getRoom gs) c ≠ none)

-- killEnemy 减小总 HP（有剑且前方有怪物时）
axiom killEnemy_totalHP_lt (s : GameState) (coord : Coord) (hSword : s.player.hasSword) (hFind : findEnemy (getRoom s) coord ≠ none) : ¬ (¬ (totalMonsterHP (killEnemy s coord) < totalMonsterHP s))

-- handleMove 后玩家坐标 = front（非墙非刺时）
axiom handleMove_player_coord (s : GameState) (d : Direction) (hNotWall : getTile (getRoom s) (front s.player.coord d) ≠ Tile.wall) (hNotSpike : getTile (getRoom s) (front s.player.coord d) ≠ Tile.spike) : ¬ (¬ ((handleMove { direction := d, action := Action.move } s d).player.coord = front s.player.coord d))

-- handleMove 不增加怪物总 HP
axiom handleMove_totalHP_nonincrease (s : GameState) (d : Direction) : ¬ (¬ (totalMonsterHP (handleMove { direction := d, action := Action.move } s d) ≤ totalMonsterHP s))

-- hasSword 在 game step 中保持不变
axiom hasSword_preserved_step (gs gs' : GameState) (input : Input) (hStep : step gs input gs') : ¬ (¬ (gs'.player.hasSword = gs.player.hasSword))

-- blockersCoverObstacles 在 game step 中保持（墙壁/陷阱布局不变）
axiom blockersCoverObstacles_preserved (gs gs' : GameState) (blockers : List Coord) (input : Input) (hCover : blockersCoverObstacles gs blockers) (hStep : step gs input gs') : ¬ (¬ (blockersCoverObstacles gs' blockers))

-- openChest 开启未开启的宝箱
axiom openChest_opens (s : GameState) (coord : Coord) (hChest : isChest (getTile (getRoom s) coord) = true) : ¬ (¬ (match getTile (getRoom s) coord with
  | Tile.chest false _ _ _ =>
    match getTile (getRoom (openChest s coord)) coord with
    | Tile.chest true _ _ _ => True
    | _ => False
  | _ => True))

-- getCurrentChests 排除已开启的宝箱
axiom getCurrentChests_excludes_opened (gs : GameState) (c : Coord) : match getTile (getRoom gs) c with
  | Tile.chest true _ _ _ => c ∉ getCurrentChests gs
  | _ => True

-- 无怪物状态在 game step 中保持不变（怪物不会凭空产生）
axiom noMonsters_preserved_step (gs gs' : GameState) (input : Input) (hStep : step gs input gs') (hNoMonsters : (getCurrentMonsters gs).isEmpty = true) : ¬ (¬ ((getCurrentMonsters gs').isEmpty = true))

-- 在有有效宝箱等待开启时，key 保持为 0（宝箱内容不包含钥匙，或开完后有效宝箱集为空）
axiom key_stays_zero_while_chests (gs gs' : GameState) (input : Input) (as : AgentState) (hStep : step gs input gs') (hKeyZero : gs.player.key = 0) (hValidChestsNE : (getCurrentChests gs).filter (fun c => !as.rememberedBlockers.contains c) ≠ []) : ¬ (¬ (gs'.player.key = 0))

-- executeObjective 始终将传入的 AgentState 作为输出的第二分量返回
axiom executeObjective_snd_eq (gs : GameState) (as : AgentState) (obj : Objective) : ¬ (¬ ((executeObjective gs as obj).snd = as))
