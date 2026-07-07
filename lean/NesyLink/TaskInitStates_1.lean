import NesyLink.Environment
open NesyLink

namespace NesyLink

-- ============================================================
-- Task 1：单房间，地图 room_001
-- 目标：走到左上宝箱拿到钥匙 → 走到上方锁着的出口通关
-- ============================================================

def task1_room : RoomLayout := {
  walls               := [
    (0, 2), (1, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2),
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)
  ],
  spawns              := [("default", (4, 6)), ("from_south", (4, 6))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (4, 0), exitType := ExitType.lockedKey, direction := ExitDirection.north,
      targetRoomId := "room_001", targetEntry := "from_south",
      requires := { keyCount := some 1, consumeKey := true }, completeTask := true : ExitInfo },
    { pos := (5, 0), exitType := ExitType.lockedKey, direction := ExitDirection.north,
      targetRoomId := "room_001", targetEntry := "from_south",
      requires := { keyCount := some 1, consumeKey := true }, completeTask := true : ExitInfo }
  ],
  initialChests       := [(0, 3)],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  roomId              := "room_001"
}

def task1_init : SymbolicState := {
  player       := (4, 6),
  health       := 5,
  keys         := 0,
  gold         := 0,
  items        := [],
  monsters     := [],
  monsterTypes := [],
  chests       := [(0, 3)],
  buttons      := [],
  room         := task1_room
}

end NesyLink
