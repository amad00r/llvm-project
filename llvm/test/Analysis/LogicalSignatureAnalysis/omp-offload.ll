; RUN: opt -passes='print<logical-signature>' -disable-output < %s 2>&1 | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: LogicalSignatureAnalysis estimates __omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388(i64, double*, double*, double*, double*, ptr)
define weak_odr protected amdgpu_kernel void @__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388(i64 noundef %n_particles, ptr noundef nonnull align 8 dereferenceable(8) %absorption, ptr noundef nonnull align 8 dereferenceable(8) %collision, ptr noundef nonnull align 8 dereferenceable(8) %tracklength, ptr noundef nonnull align 8 dereferenceable(8) %leakage, ptr noalias noundef %dyn_ptr) local_unnamed_addr #0 {
entry:
  %0 = addrspacecast ptr %leakage to ptr addrspace(1)
  %1 = addrspacecast ptr %tracklength to ptr addrspace(1)
  %2 = addrspacecast ptr %collision to ptr addrspace(1)
  %3 = addrspacecast ptr %absorption to ptr addrspace(1)
  %absorption1.i = alloca double, align 8, addrspace(5)
  %collision2.i = alloca double, align 8, addrspace(5)
  %tracklength3.i = alloca double, align 8, addrspace(5)
  %leakage4.i = alloca double, align 8, addrspace(5)
  %captured_vars_addrs.i = alloca [7 x ptr], align 8, addrspace(5)
  %.omp.reduction.red_list.i = alloca [4 x ptr], align 8, addrspace(5)
  %4 = tail call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_kernel_environment to ptr), ptr %dyn_ptr) #3
  %exec_user_code = icmp eq i32 %4, -1
  br i1 %exec_user_code, label %user_code.entry, label %common.ret

common.ret:                                       ; preds = %entry, %__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_omp_outlined.exit
  ret void

user_code.entry:                                  ; preds = %entry
  %5 = tail call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @1 to ptr)) #3
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %captured_vars_addrs.i)
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %.omp.reduction.red_list.i)
  %.omp.reduction.red_list.ascast.i = addrspacecast ptr addrspace(5) %.omp.reduction.red_list.i to ptr
  %absorption1.ascast.i = addrspacecast ptr addrspace(5) %absorption1.i to ptr
  %collision2.ascast.i = addrspacecast ptr addrspace(5) %collision2.i to ptr
  %tracklength3.ascast.i = addrspacecast ptr addrspace(5) %tracklength3.i to ptr
  %leakage4.ascast.i = addrspacecast ptr addrspace(5) %leakage4.i to ptr
  %n_particles.addr.sroa.0.0.extract.trunc.i = trunc i64 %n_particles to i32
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %absorption1.i) #54, !noalias !744
  store double 0.000000e+00, ptr addrspace(5) %absorption1.i, align 8, !tbaa !166, !noalias !744
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %collision2.i) #54, !noalias !744
  store double 0.000000e+00, ptr addrspace(5) %collision2.i, align 8, !tbaa !166, !noalias !744
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %tracklength3.i) #54, !noalias !744
  store double 0.000000e+00, ptr addrspace(5) %tracklength3.i, align 8, !tbaa !166, !noalias !744
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %leakage4.i) #54, !noalias !744
  store double 0.000000e+00, ptr addrspace(5) %leakage4.i, align 8, !tbaa !166, !noalias !744
  %cmp.i = icmp sgt i32 %n_particles.addr.sroa.0.0.extract.trunc.i, 0
  br i1 %cmp.i, label %omp.precond.then.i, label %omp.precond.end.i

omp.precond.then.i:                               ; preds = %user_code.entry
  %n_particles.casted.sroa.0.0.insert.ext = and i64 %n_particles, 2147483647
  %sub6.i = add i64 %n_particles, 4294967295
  %captured_vars_addrs.ascast.i = addrspacecast ptr addrspace(5) %captured_vars_addrs.i to ptr
  %6 = and i64 %sub6.i, 4294967295
  store ptr null, ptr addrspace(5) %captured_vars_addrs.i, align 8, !tbaa !130, !noalias !744
  %7 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 8
  %8 = inttoptr i64 %6 to ptr
  store ptr %8, ptr addrspace(5) %7, align 8, !tbaa !130, !noalias !744
  %9 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 16
  %10 = inttoptr i64 %n_particles.casted.sroa.0.0.insert.ext to ptr
  store ptr %10, ptr addrspace(5) %9, align 8, !tbaa !130, !noalias !744
  %11 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 24
  store ptr %absorption1.ascast.i, ptr addrspace(5) %11, align 8, !tbaa !130, !noalias !744
  %12 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 32
  store ptr %collision2.ascast.i, ptr addrspace(5) %12, align 8, !tbaa !130, !noalias !744
  %13 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 40
  store ptr %tracklength3.ascast.i, ptr addrspace(5) %13, align 8, !tbaa !130, !noalias !744
  %14 = getelementptr inbounds nuw i8, ptr addrspace(5) %captured_vars_addrs.i, i32 48
  store ptr %leakage4.ascast.i, ptr addrspace(5) %14, align 8, !tbaa !130, !noalias !744
  call void @__kmpc_parallel_60(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i32 %5, i32 1, i32 -1, i32 -1, ptr nonnull @__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_omp_outlined_omp_outlined, ptr null, ptr %captured_vars_addrs.ascast.i, i64 7, i32 0) #3, !noalias !744
  br label %omp.precond.end.i

omp.precond.end.i:                                ; preds = %omp.precond.then.i, %user_code.entry
  store ptr %absorption1.ascast.i, ptr addrspace(5) %.omp.reduction.red_list.i, align 8, !noalias !744
  %15 = getelementptr inbounds nuw i8, ptr addrspace(5) %.omp.reduction.red_list.i, i32 8
  store ptr %collision2.ascast.i, ptr addrspace(5) %15, align 8, !noalias !744
  %16 = getelementptr inbounds nuw i8, ptr addrspace(5) %.omp.reduction.red_list.i, i32 16
  store ptr %tracklength3.ascast.i, ptr addrspace(5) %16, align 8, !noalias !744
  %17 = getelementptr inbounds nuw i8, ptr addrspace(5) %.omp.reduction.red_list.i, i32 24
  store ptr %leakage4.ascast.i, ptr addrspace(5) %17, align 8, !noalias !744
  %18 = call i32 @__kmpc_gpu_xteam_reduce_nowait(ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr %.omp.reduction.red_list.ascast.i, ptr nonnull @_omp_reduction_shuffle_and_reduce_func.12, ptr nonnull @_omp_reduction_inter_warp_copy_func.13, ptr nonnull @_omp_reduction_list_to_global_copy_func.14, ptr nonnull @_omp_reduction_global_to_list_copy_func.15, ptr nonnull @_omp_reduction_global_to_list_reduce_func.16) #3, !noalias !744
  %19 = icmp eq i32 %18, 1
  br i1 %19, label %.omp.reduction.then.i, label %__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_omp_outlined.exit

.omp.reduction.then.i:                            ; preds = %omp.precond.end.i
  %20 = load double, ptr addrspace(1) %3, align 8, !tbaa !166, !noalias !744
  %21 = load double, ptr addrspace(5) %absorption1.i, align 8, !tbaa !166, !noalias !744
  %add.i = fadd double %20, %21
  store double %add.i, ptr addrspace(1) %3, align 8, !tbaa !166, !noalias !744
  %22 = load double, ptr addrspace(1) %2, align 8, !tbaa !166, !noalias !744
  %23 = load double, ptr addrspace(5) %collision2.i, align 8, !tbaa !166, !noalias !744
  %add8.i = fadd double %22, %23
  store double %add8.i, ptr addrspace(1) %2, align 8, !tbaa !166, !noalias !744
  %24 = load double, ptr addrspace(1) %1, align 8, !tbaa !166, !noalias !744
  %25 = load double, ptr addrspace(5) %tracklength3.i, align 8, !tbaa !166, !noalias !744
  %add9.i = fadd double %24, %25
  store double %add9.i, ptr addrspace(1) %1, align 8, !tbaa !166, !noalias !744
  %26 = load double, ptr addrspace(1) %0, align 8, !tbaa !166, !noalias !744
  %27 = load double, ptr addrspace(5) %leakage4.i, align 8, !tbaa !166, !noalias !744
  %add10.i = fadd double %26, %27
  store double %add10.i, ptr addrspace(1) %0, align 8, !tbaa !166, !noalias !744
  br label %__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_omp_outlined.exit

__omp_offloading_53_25e07d0__ZN6openmc20process_death_eventsEi_l388_omp_outlined.exit: ; preds = %omp.precond.end.i, %.omp.reduction.then.i
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %leakage4.i) #3, !noalias !744
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %tracklength3.i) #3, !noalias !744
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %collision2.i) #3, !noalias !744
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %absorption1.i) #3, !noalias !744
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %captured_vars_addrs.i)
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %.omp.reduction.red_list.i)
  call void @__kmpc_target_deinit() #3
  br label %common.ret
}
