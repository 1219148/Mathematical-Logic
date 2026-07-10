import NesyLink.Environment
open NesyLink

namespace NesyLink

-- Task 2：单房间
-- 目标：避开陷阱，击杀怪物，拿到钥匙，满足条件出口

def task2_trapPositions : List Position := [
  (1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),
  (1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7)
]

def task2_room : RoomLayout := {
  walls               := [],
  spawns              := [("default", (7, 3)), ("from_east", (8, 4))],
  defaultSpawn        := "default",
  traps               := task2_trapPositions.map (fun p =>
    { pos := p, damage := 1, respawnTo := "default" : TrapInfo }),
  exitInfos           := [
    { pos := (0, 3), exitType := ExitType.conditional, direction := ExitDirection.west,
      targetRoomId := "room_001", targetEntry := "from_east",
      requires := { keyCount := some 1, allMonstersDefeated := true },
      completeTask := true : ExitInfo },
    { pos := (0, 4), exitType := ExitType.conditional, direction := ExitDirection.west,
      targetRoomId := "room_001", targetEntry := "from_east",
      requires := { keyCount := some 1, allMonstersDefeated := true },
      completeTask := true : ExitInfo }
  ],
  initialChests       := [(1, 3)],
  initialMonsters     := [(2, 2)],
  initialMonsterTypes := [((2, 2), MonsterType.chaser)],
  initialButtons      := [],
  roomId              := "room_001"
}

def task2_init : SymbolicState := {
  player       := (7, 3),
  facing       := Direction.left,
  health       := 5,
  keys         := 0,
  gold         := 0,
  items        := [],
  monsters     := [{ damage := 1, hp := 3, pos := (2, 2), monsterType := MonsterType.chaser : MonsterInfo }],
  monsterTypes := [((2, 2), MonsterType.chaser)],
  chests       := [{ pos := (1, 3), opened := false : ChestInfo }],
  buttons      := [],
  shieldTicks := 0,
  room         := task2_room
}

end NesyLink
