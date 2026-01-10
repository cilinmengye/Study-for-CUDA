# Makefile
NVCC = nvcc

ARCH = -arch=sm_89

CFLAGS = -O3 -lineinfo $(ARCH) \
         --expt-relaxed-constexpr \
         --extended-lambda \
         --forward-unknown-to-host-compiler \
         -D_FORCE_INLINES \
         -D_MWAITXINTRIN_H_INCLUDED \
         -D__STRICT_ANSI__ \
         -D__USE_MISC \
         -D_GNU_SOURCE \
         -isystem=/usr/local/cuda/include

ORIGINAL = matmul_original
ORIGINAL_PERF = matmul_original_perf
TT = matmul_tt
TT_PERF = matmul_tt_perf

$(ORIGINAL): main.cu warptiling.cuh
	$(NVCC) $(CFLAGS) -include warptiling.cuh main.cu -o $(ORIGINAL)

$(ORIGINAL_PERF): perf.cu warptiling.cuh
	$(NVCC) $(CFLAGS) -include warptiling.cuh perf.cu -o $(ORIGINAL_PERF)

$(TT): maintt.cu warptiling_threadtiling.cuh
	$(NVCC) $(CFLAGS) -include warptiling_threadtiling.cuh maintt.cu -o $(TT)

$(TT_PERF): perftt.cu warptiling_threadtiling.cuh
	$(NVCC) $(CFLAGS) -include warptiling_threadtiling.cuh perftt.cu -o $(TT_PERF)

clean:
	rm -f $(ORIGINAL) $(ORIGINAL_PERF)
	rm -f $(TT) $(TT_PERF)

run_org: $(ORIGINAL)
	./$(ORIGINAL)

run_tt: $(TT)
	./$(TT)

profile_ct_org: $(ORIGINAL_PERF)
	./$(ORIGINAL_PERF)

profile_ct_tt: $(TT_PERF)
	./$(TT_PERF)

profile_org: $(ORIGINAL)
	ncu --set full --kernel-name "sgemmWarptiling" -o original.ncu-rep $(ORIGINAL)

profile_tt: $(TT)
	ncu --set full --kernel-name "sgemmWarptiling" -o tt.ncu-rep $(TT)