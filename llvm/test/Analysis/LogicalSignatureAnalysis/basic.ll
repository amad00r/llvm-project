; RUN: opt -passes='print<logical-signature>' -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: LogicalSignatureAnalysis estimates unused(ptr, i64)
define void @unused(ptr %p, i64 %x) {
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates store_load(float*, float, i16, i16*)
define void @store_load(ptr %0, i64 %1, i64 %2, ptr %3) {
  %5 = alloca ptr
  %6 = alloca i64
  %7 = alloca i64
  %8 = alloca ptr
  store ptr %0, ptr %5
  store i64 %1, ptr %6
  store i64 %2, ptr %7
  store ptr %3, ptr %8
  %9 = load ptr, ptr %5
  %10 = load float, ptr %6
  %11 = load i16, ptr %7
  %12 = load ptr, ptr %8
  store float %10, ptr %9
  store i16 %11, ptr %12
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates interprocedural(float*, float, i16, i16*)
define void @interprocedural(ptr %0, i64 %1, i64 %2, ptr %3) {
  call void @store_load(ptr %0, i64 %1, i64 %2, ptr %3)
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates bitcast_scalar(float*, float)
define void @bitcast_scalar(ptr %out, i32 %bits) {
  %as_float = bitcast i32 %bits to float
  store float %as_float, ptr %out
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates trunc_scalar(i16*, i16)
define void @trunc_scalar(ptr %out, i64 %wide) {
  %narrow = trunc i64 %wide to i16
  store i16 %narrow, ptr %out
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates addrspacecast_pointer(double*, double*)
define void @addrspacecast_pointer(ptr addrspace(1) %src, ptr %dst) {
  %generic = addrspacecast ptr addrspace(1) %src to ptr
  %value = load double, ptr %generic
  store double %value, ptr %dst
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates cast_chain(i8*, i8)
define void @cast_chain(ptr addrspace(1) %out.as1, i64 %wide) {
  %out = addrspacecast ptr addrspace(1) %out.as1 to ptr
  %narrow = trunc i64 %wide to i8
  store i8 %narrow, ptr %out
  ret void
}

; CHECK-LABEL: LogicalSignatureAnalysis estimates conflicting_uses(ptr)
define void @conflicting_uses(ptr %out) {
  store i32 0, ptr %out
  store float 0.000000e+00, ptr %out
  ret void
}
