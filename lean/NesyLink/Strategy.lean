import NesyLink.Environment
import NesyLink.TaskInitStates
open NesyLink

namespace NesyLink

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | refl (s : SymbolicState) : Exec s [] s
  | step (s t u : SymbolicState) (a : Action) (plan : List Action) :
      Step s a t → Exec t plan u → Exec s (a :: plan) u

inductive GoalType where
  | reachExit
  | killAllMonsters
  | openAllChests
  deriving DecidableEq, Repr


-- BFS形式化和可达性

-- 策略性质

end NesyLink
