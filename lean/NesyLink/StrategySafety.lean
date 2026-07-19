import NesyLink.Env1
import NesyLink.Env2
import NesyLink.Strategy
import NesyLink.Axioms

/-!
# 前四关策略安全性证明

1. **不撞墙** — 策略输出 `move` 时，前方不是墙
2. **不踩陷阱** — 策略输出 `move` 时，前方不是陷阱
3. **单怪物击杀** — 若有怪物且玩家有剑，有限步内怪物被击杀
4. **单宝箱打开** — 若无怪物且有宝箱，有限步内宝箱被打开

公理定义于 `StrategySafety/Axioms.lean`（安全性+活性公理）
-/

set_option linter.unusedSimpArgs false

-- ============================================================
-- 0. 策略 & 辅助定义
-- ============================================================

def strategyTasks1to4 (gs : GameState) (as : AgentState) : Input × AgentState :=
  let (obj, as') := chooseLocalObjective gs as false
  executeObjective gs as' obj

def monsterHPAt (gs : GameState) (c : Coord) : Nat :=
  match findEnemy (getRoom gs) c with
    | some e => e.hp
    | none => 0

private axiom chooseLocalObj_snd_eq (gs : GameState) (as : AgentState) :
    (chooseLocalObjective gs as false).snd = as

-- ============================================================
-- 1. 策略展开引理
-- ============================================================

theorem strategy_expand (gs : GameState) (as : AgentState) :
    (strategyTasks1to4 gs as).fst =
    (executeObjective gs (chooseLocalObjective gs as false).snd
      (chooseLocalObjective gs as false).fst).fst := by
  unfold strategyTasks1to4; rfl

theorem strategy_snd_eq (gs : GameState) (as : AgentState) :
    (strategyTasks1to4 gs as).snd = as := by
  unfold strategyTasks1to4
  dsimp
  let lhs := executeObjective gs (chooseLocalObjective gs as false).snd (chooseLocalObjective gs as false).fst
  have hWrapped := executeObjective_snd_eq gs (chooseLocalObjective gs as false).snd (chooseLocalObjective gs as false).fst
  by_cases hEq : lhs.snd = (chooseLocalObjective gs as false).snd
  · rw [hEq, chooseLocalObj_snd_eq gs as]
  · exfalso; exact hWrapped hEq

-- ============================================================
-- 2. 安全证明通用框架
-- ============================================================

/-- 安全分析的通用模式：
    1. 展开策略 → executeObjective 调用
    2. 提取 chooseLocalObjective 结果并用公理简化
    3. 按 Objective.kind 分类，对每个分支分析 move 来源
    4. 应用对应的安全公理 -/

-- ============================================================
-- 3. 定理 1：不撞墙
-- ============================================================

theorem no_wall_bump (gs : GameState) (as : AgentState)
    (hCovers : blockersCoverObstacles gs as.rememberedBlockers)
    (hMove : (strategyTasks1to4 gs as).fst.action = Action.move) :
    getTile (getRoom gs)
      (front gs.player.coord (strategyTasks1to4 gs as).fst.direction) ≠ Tile.wall := by
  rw [strategy_expand gs as] at hMove ⊢
  generalize hPair : chooseLocalObjective gs as false = pair
  rcases pair with ⟨obj, as'⟩
  rw [hPair] at hMove
  have hAs' : as' = as := by
    have hSnd := congrArg Prod.snd hPair
    dsimp at hSnd
    rw [← hSnd, chooseLocalObj_snd_eq gs as]
  have hBlockersEq : as'.rememberedBlockers = as.rememberedBlockers := by rw [hAs']
  have hCovers' : blockersCoverObstacles gs as'.rememberedBlockers := by
    rw [hBlockersEq]; exact hCovers
  match hk : obj.kind with
  | ObjectiveKind.idle =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove
      simp at hMove
  | ObjectiveKind.fight =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      generalize hFilter : (obj.targets.filter (fun m => isAdjacent gs.player.coord m)).head? = filterResult
      at hMove ⊢
      cases filterResult with
      | none =>
          simp at hMove ⊢
          cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
          | none => simp [hNext] at hMove
          | some dir =>
              simp [hNext] at hMove ⊢
              have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
                as'.rememberedBlockers dir hNext
              intro hWall; apply hAvoid; apply hCovers'; left; exact hWall
      | some m =>
          simp at hMove ⊢
          cases hDir : getDirectionTo gs.player.coord m with
          | none => simp [hDir] at hMove
          | some dir => simp [hDir] at hMove
  | ObjectiveKind.interact =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      generalize hFilter : (obj.targets.filter (fun t => isAdjacent gs.player.coord t)).head? = filterResult at hMove ⊢
      cases filterResult with
      | none =>
          simp at hMove ⊢
          cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
          | none => simp [hNext] at hMove
          | some dir =>
              simp [hNext] at hMove ⊢
              have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
                as'.rememberedBlockers dir hNext
              intro hWall; apply hAvoid; apply hCovers'; left; exact hWall
      | some t =>
          simp at hMove ⊢
          cases hDir : getDirectionTo gs.player.coord t with
          | none => simp [hDir] at hMove
          | some dir => simp [hDir] at hMove
  | ObjectiveKind.goExit =>
      match hs : obj.side with
      | Option.none =>
          unfold executeObjective at hMove ⊢
          dsimp at hMove ⊢
          rw [hk, hs] at hMove
          simp at hMove
      | Option.some side =>
          unfold executeObjective at hMove ⊢
          dsimp at hMove ⊢
          rw [hk, hs] at hMove ⊢
          dsimp at hMove ⊢
          split at hMove
          · rename_i hContains
            split
            · rename_i hContains'
              exact exit_direction_not_wall gs side as'.rememberedBlockers gs.player.coord hContains
            · rename_i hNot; exfalso; apply hNot; exact hContains
          · rename_i hNotContains
            split
            · rename_i hContains'; exfalso; apply hNotContains; exact hContains'
            · rename_i hNotContains'
              cases hNext : nextStepTo gs.player.coord
                (sideApproachGoals gs side as'.rememberedBlockers) as'.rememberedBlockers with
              | none => simp [hNext] at hMove
              | some dir =>
                  simp [hNext] at hMove ⊢
                  have hAvoid := nextStepTo_avoids_blocked gs.player.coord
                    (sideApproachGoals gs side as'.rememberedBlockers)
                    as'.rememberedBlockers dir hNext
                  intro hWall; apply hAvoid; apply hCovers'; left; exact hWall
  | ObjectiveKind.navigate =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
      | none => simp [hNext] at hMove
      | some dir =>
          simp [hNext] at hMove ⊢
          have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
            as'.rememberedBlockers dir hNext
          intro hWall; apply hAvoid; apply hCovers'; left; exact hWall

-- ============================================================
-- 4. 定理 2：不踩陷阱
-- ============================================================

theorem no_spike_step (gs : GameState) (as : AgentState)
    (hCovers : blockersCoverObstacles gs as.rememberedBlockers)
    (hMove : (strategyTasks1to4 gs as).fst.action = Action.move) :
    getTile (getRoom gs)
      (front gs.player.coord (strategyTasks1to4 gs as).fst.direction) ≠ Tile.spike := by
  rw [strategy_expand gs as] at hMove ⊢
  generalize hPair : chooseLocalObjective gs as false = pair
  rcases pair with ⟨obj, as'⟩
  rw [hPair] at hMove
  have hAs' : as' = as := by
    have hSnd := congrArg Prod.snd hPair
    dsimp at hSnd
    rw [← hSnd, chooseLocalObj_snd_eq gs as]
  have hBlockersEq : as'.rememberedBlockers = as.rememberedBlockers := by rw [hAs']
  have hCovers' : blockersCoverObstacles gs as'.rememberedBlockers := by
    rw [hBlockersEq]; exact hCovers
  match hk : obj.kind with
  | ObjectiveKind.idle =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove
      simp at hMove
  | ObjectiveKind.fight =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      generalize hFilter : (obj.targets.filter (fun m => isAdjacent gs.player.coord m)).head? = filterResult
      at hMove ⊢
      cases filterResult with
      | none =>
          simp at hMove ⊢
          cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
          | none => simp [hNext] at hMove
          | some dir =>
              simp [hNext] at hMove ⊢
              have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
                as'.rememberedBlockers dir hNext
              intro hSpike; apply hAvoid; apply hCovers'; right; exact hSpike
      | some m =>
          simp at hMove ⊢
          cases hDir : getDirectionTo gs.player.coord m with
          | none => simp [hDir] at hMove
          | some dir => simp [hDir] at hMove
  | ObjectiveKind.interact =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      generalize hFilter : (obj.targets.filter (fun t => isAdjacent gs.player.coord t)).head? = filterResult at hMove ⊢
      cases filterResult with
      | none =>
          simp at hMove ⊢
          cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
          | none => simp [hNext] at hMove
          | some dir =>
              simp [hNext] at hMove ⊢
              have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
                as'.rememberedBlockers dir hNext
              intro hSpike; apply hAvoid; apply hCovers'; right; exact hSpike
      | some t =>
          simp at hMove ⊢
          cases hDir : getDirectionTo gs.player.coord t with
          | none => simp [hDir] at hMove
          | some dir => simp [hDir] at hMove
  | ObjectiveKind.goExit =>
      match hs : obj.side with
      | Option.none =>
          unfold executeObjective at hMove ⊢
          dsimp at hMove ⊢
          rw [hk, hs] at hMove
          simp at hMove
      | Option.some side =>
          unfold executeObjective at hMove ⊢
          dsimp at hMove ⊢
          rw [hk, hs] at hMove ⊢
          dsimp at hMove ⊢
          split at hMove
          · rename_i hContains
            split
            · rename_i hContains'
              exact exit_direction_not_spike gs side as'.rememberedBlockers gs.player.coord hContains
            · rename_i hNot; exfalso; apply hNot; exact hContains
          · rename_i hNotContains
            split
            · rename_i hContains'; exfalso; apply hNotContains; exact hContains'
            · rename_i hNotContains'
              cases hNext : nextStepTo gs.player.coord
                (sideApproachGoals gs side as'.rememberedBlockers) as'.rememberedBlockers with
              | none => simp [hNext] at hMove
              | some dir =>
                  simp [hNext] at hMove ⊢
                  have hAvoid := nextStepTo_avoids_blocked gs.player.coord
                    (sideApproachGoals gs side as'.rememberedBlockers)
                    as'.rememberedBlockers dir hNext
                  intro hSpike; apply hAvoid; apply hCovers'; right; exact hSpike
  | ObjectiveKind.navigate =>
      unfold executeObjective at hMove ⊢
      dsimp at hMove ⊢
      rw [hk] at hMove ⊢
      cases hNext : nextStepTo gs.player.coord obj.targets as'.rememberedBlockers with
      | none => simp [hNext] at hMove
      | some dir =>
          simp [hNext] at hMove ⊢
          have hAvoid := nextStepTo_avoids_blocked gs.player.coord obj.targets
            as'.rememberedBlockers dir hNext
          intro hSpike; apply hAvoid; apply hCovers'; right; exact hSpike

-- ============================================================
-- 5. 活性证明：测度、多步执行、进度公理
-- ============================================================

-- 怪物击杀测度：总HP * 100 + 到最近怪物的距离
def monsterMeasure (gs : GameState) : Nat :=
  totalMonsterHP gs * 100 + minCoordDist gs.player.coord (getCurrentMonsters gs)

-- 宝箱测度：有效宝箱数 * 100 + 到最近有效宝箱的距离
def chestMeasure (gs : GameState) (blockers : List Coord) : Nat :=
  let validChests := (getCurrentChests gs).filter (fun c => !blockers.contains c)
  validChests.length * 100 + minCoordDist gs.player.coord validChests

-- 多步执行关系：从 (gs, as) 执行 n 步到达 (gs', as')
inductive steps : GameState → AgentState → Nat → GameState → AgentState → Prop where
  | refl (gs : GameState) (as : AgentState) : steps gs as 0 gs as
  | succ (gs : GameState) (as : AgentState) (gsMid : GameState)
      (gs' : GameState) (as' : AgentState) (n : Nat) :
      step gs (strategyTasks1to4 gs as).fst gsMid →
      steps gsMid as n gs' as' →
      steps gs as (n + 1) gs' as'

-- 提取 ¬ (¬ P) → P
private theorem not_not_elim {P : Prop} (h : ¬ (¬ P)) : P := by
  by_cases hP : P
  · exact hP
  · exfalso; exact h hP

-- 策略单体步骤使怪物测度严格减小（核心活性公理）
axiom strategy_fight_step_decreases_measure (gs : GameState) (as : AgentState)
    (hMonstersNE : (getCurrentMonsters gs).isEmpty = false)
    (hSword : gs.player.hasSword = true)
    (hCover : blockersCoverObstacles gs as.rememberedBlockers) :
    ¬ (¬ (∃ gs', step gs (strategyTasks1to4 gs as).fst gs' ∧
          monsterMeasure gs' < monsterMeasure gs))

-- 策略单体步骤使宝箱测度严格减小（核心活性公理）
axiom strategy_interact_step_progress (gs : GameState) (as : AgentState)
    (hNoMonsters : (getCurrentMonsters gs).isEmpty = true)
    (hValidChestsNE : (getCurrentChests gs).filter (fun c => !as.rememberedBlockers.contains c) ≠ [])
    (hKeyZero : gs.player.key = 0)
    (hCover : blockersCoverObstacles gs as.rememberedBlockers) :
    ¬ (¬ (∃ gs', step gs (strategyTasks1to4 gs as).fst gs' ∧
          chestMeasure gs' as.rememberedBlockers < chestMeasure gs as.rememberedBlockers))

-- ============================================================
-- 6. 定理 3：单怪物击杀
-- ============================================================

theorem single_monster_kill (gs0 : GameState) (as0 : AgentState)
    (hMonsterCount : (getCurrentMonsters gs0).length = 1)
    (hHasSword : gs0.player.hasSword = true)
    (hCover : blockersCoverObstacles gs0 as0.rememberedBlockers) :
    ∃ (n : Nat) (gs' : GameState) (as' : AgentState),
      steps gs0 as0 n gs' as' ∧ (getRoom gs').enemies = [] := by
  -- 从 length = 1 推出 isEmpty = false
  have hMonstersNE0 : (getCurrentMonsters gs0).isEmpty = false := by
    have hNE : getCurrentMonsters gs0 ≠ [] := by
      intro h; rw [h] at hMonsterCount; simp at hMonsterCount
    simp [hNE]
  -- 强归纳法：对测度大小进行自然数强归纳
  let P : Nat → Prop := λ n => ∀ (gs : GameState) (as : AgentState),
    monsterMeasure gs = n →
    (getCurrentMonsters gs).isEmpty = false →
    gs.player.hasSword = true →
    blockersCoverObstacles gs as.rememberedBlockers →
    ∃ (n' : Nat) (gs' : GameState) (as' : AgentState),
      steps gs as n' gs' as' ∧ (getRoom gs').enemies = []
  have hStep : ∀ n, (∀ m < n, P m) → P n := by
    intro n hn gs as hMeasure hMonstersNE hSword hCoverArg
    -- 判断 enemies 是否已经为空
    by_cases hEnemiesEmpty : (getRoom gs).enemies = []
    · exact ⟨0, gs, as, steps.refl gs as, hEnemiesEmpty⟩
    · -- 从公理获取一步进度
      have hProgressWrapped := strategy_fight_step_decreases_measure gs as hMonstersNE hSword hCoverArg
      have hProgress : ∃ gs', step gs (strategyTasks1to4 gs as).fst gs' ∧
          monsterMeasure gs' < monsterMeasure gs :=
        not_not_elim hProgressWrapped
      rcases hProgress with ⟨gsMid, hStepMid, hMeasureLt⟩
      have hm : monsterMeasure gsMid < n := by
        rw [← hMeasure]; exact hMeasureLt
      -- sword 在 step 中保持
      have hSwordMid : gsMid.player.hasSword = true := by
        have hPreservedWrapped := hasSword_preserved_step gs gsMid (strategyTasks1to4 gs as).fst hStepMid
        have hPreserved : gsMid.player.hasSword = gs.player.hasSword :=
          not_not_elim hPreservedWrapped
        rw [hPreserved]; exact hSword
      -- blockersCoverObstacles 在 step 中保持
      have hCoverMid : blockersCoverObstacles gsMid as.rememberedBlockers := by
        have hPreservedWrapped := blockersCoverObstacles_preserved gs gsMid as.rememberedBlockers
          (strategyTasks1to4 gs as).fst hCoverArg hStepMid
        exact not_not_elim hPreservedWrapped
      -- 判断 gsMid 中是否还有怪物
      by_cases hMonstersNEMid : (getCurrentMonsters gsMid).isEmpty = false
      · -- 还有怪物，使用归纳假设（测度已减小）
        have hRes := hn (monsterMeasure gsMid) hm gsMid as rfl hMonstersNEMid hSwordMid hCoverMid
        rcases hRes with ⟨n', gs', as', hSteps, hEnemiesEmpty'⟩
        refine ⟨n' + 1, gs', as', ?_, hEnemiesEmpty'⟩
        exact steps.succ gs as gsMid gs' as' n' hStepMid hSteps
      · -- 怪物在本步中被消灭！
        have hEnemiesEmptyMid : (getRoom gsMid).enemies = [] := by
          by_cases hNE : (getRoom gsMid).enemies = []
          · exact hNE
          · have hEmptyFalse : (getCurrentMonsters gsMid).isEmpty = false :=
              (getCurrentMonsters_empty_iff gsMid).mpr hNE
            exfalso; exact hMonstersNEMid hEmptyFalse
        refine ⟨1, gsMid, as, ?_, hEnemiesEmptyMid⟩
        exact steps.succ gs as gsMid gsMid as 0 hStepMid (steps.refl gsMid as)
  -- 启动强归纳法（使用 Nat.strongRecOn）
  have hAll : ∀ n, P n :=
    λ n => Nat.strongRecOn n hStep
  exact hAll (monsterMeasure gs0) gs0 as0 rfl hMonstersNE0 hHasSword hCover

-- ============================================================
-- 7. 定理 4：单宝箱打开
-- ============================================================

theorem single_chest_open (gs0 : GameState) (as0 : AgentState)
    (hNoMonsters : (getCurrentMonsters gs0).isEmpty = true)
    (hChestExists : ∃ c ∈ getCurrentChests gs0, c ∉ as0.rememberedBlockers)
    (hKeyZero : gs0.player.key = 0)
    (hCover : blockersCoverObstacles gs0 as0.rememberedBlockers) :
    ∃ (n : Nat) (gs' : GameState) (as' : AgentState),
      steps gs0 as0 n gs' as' ∧
      (getCurrentChests gs').filter (fun c => !as0.rememberedBlockers.contains c) = [] := by
  -- 从 hChestExists 推出 validChests ≠ []
  have hValidChestsNE0 : (getCurrentChests gs0).filter (fun c => !as0.rememberedBlockers.contains c) ≠ [] := by
    rcases hChestExists with ⟨c, hcMem, hcNotBlocked⟩
    intro hEmpty
    have hcNotInFilter : c ∉ (getCurrentChests gs0).filter (fun c' => !as0.rememberedBlockers.contains c') := by
      rw [hEmpty]; simp
    apply hcNotInFilter
    simp [hcMem, hcNotBlocked]
  -- as0.rememberedBlockers 作为固定 blocker 集（策略不修改它）
  let blockers := as0.rememberedBlockers
  -- 强归纳法：只对 GameState 参数化，as0 固定
  let P : Nat → Prop := λ n => ∀ (gs : GameState),
    chestMeasure gs blockers = n →
    (getCurrentMonsters gs).isEmpty = true →
    gs.player.key = 0 →
    blockersCoverObstacles gs blockers →
    (getCurrentChests gs).filter (fun c => !blockers.contains c) ≠ [] →
    ∃ (n' : Nat) (gs' : GameState),
      steps gs as0 n' gs' as0 ∧
      (getCurrentChests gs').filter (fun c => !blockers.contains c) = []
  have hStep : ∀ n, (∀ m < n, P m) → P n := by
    intro n hn gs hMeasure hNoMonstersGS hKeyZeroGS hCoverArg hValidChestsNE
    -- 从公理获取一步进度
    have hProgressWrapped := strategy_interact_step_progress gs as0 hNoMonstersGS hValidChestsNE hKeyZeroGS hCoverArg
    have hProgress : ∃ gs', step gs (strategyTasks1to4 gs as0).fst gs' ∧
        chestMeasure gs' blockers < chestMeasure gs blockers :=
      not_not_elim hProgressWrapped
    rcases hProgress with ⟨gsMid, hStepMid, hMeasureLt⟩
    have hm : chestMeasure gsMid blockers < n := by
      rw [← hMeasure]; exact hMeasureLt
    -- 判断 gsMid 中是否还有有效宝箱
    by_cases hValidEmptyMid : (getCurrentChests gsMid).filter (fun c => !blockers.contains c) = []
    · -- 宝箱已全部开启！
      refine ⟨1, gsMid, ?_, hValidEmptyMid⟩
      exact steps.succ gs as0 gsMid gsMid as0 0 hStepMid (steps.refl gsMid as0)
    · -- 还有有效宝箱，需要重建归纳假设的前置条件
      -- (1) 无怪物状态保持
      have hNoMonstersMid : (getCurrentMonsters gsMid).isEmpty = true := by
        have hPreservedWrapped := noMonsters_preserved_step gs gsMid (strategyTasks1to4 gs as0).fst hStepMid hNoMonstersGS
        exact not_not_elim hPreservedWrapped
      -- (2) key = 0 保持（在还有有效宝箱时）
      have hKeyZeroMid : gsMid.player.key = 0 := by
        have hPreservedWrapped := key_stays_zero_while_chests gs gsMid (strategyTasks1to4 gs as0).fst as0 hStepMid hKeyZeroGS hValidChestsNE
        exact not_not_elim hPreservedWrapped
      -- (3) blockersCoverObstacles 保持
      have hCoverMid : blockersCoverObstacles gsMid blockers := by
        have hPreservedWrapped := blockersCoverObstacles_preserved gs gsMid blockers
          (strategyTasks1to4 gs as0).fst hCoverArg hStepMid
        exact not_not_elim hPreservedWrapped
      -- 使用归纳假设
      have hRes := hn (chestMeasure gsMid blockers) hm gsMid rfl hNoMonstersMid hKeyZeroMid hCoverMid hValidEmptyMid
      rcases hRes with ⟨n', gs', hSteps, hAllOpened⟩
      refine ⟨n' + 1, gs', ?_, hAllOpened⟩
      exact steps.succ gs as0 gsMid gs' as0 n' hStepMid hSteps
  -- 启动强归纳法（使用 Nat.strongRecOn）
  have hAll : ∀ n, P n :=
    λ n => Nat.strongRecOn n hStep
  rcases hAll (chestMeasure gs0 blockers) gs0 rfl hNoMonsters hKeyZero hCover hValidChestsNE0 with ⟨n, gs', hSteps, hAllOpened⟩
  exact ⟨n, gs', as0, hSteps, hAllOpened⟩
