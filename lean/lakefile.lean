import Lake
open Lake DSL

package «nesylink-lean» where
  leanVersion := "v4.29.0-rc6"

@[default_target]
lean_lib NesyLink where
  -- 源文件位于 NesyLink/ 目录下
