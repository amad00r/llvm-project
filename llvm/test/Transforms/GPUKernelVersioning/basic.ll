; TODO: 

; RUN: opt -passes=gpu-kernel-versioning -S < %s | FileCheck %s

; CHECK: define void @kernel()
define void @kernel() {
  ret void
}
