# Makefile
NVCC = nvcc

ARCH = -arch=sm_89

CFLAGS = -O3 -lineinfo $(ARCH) \
         --use_fast_math \
         --expt-relaxed-constexpr \
         --extended-lambda \
         --forward-unknown-to-host-compiler \
         -D_FORCE_INLINES \
         -D_MWAITXINTRIN_H_INCLUDED \
         -D__STRICT_ANSI__ \
         -D__USE_MISC \
         -D_GNU_SOURCE \
         -isystem=/usr/local/cuda/include


GGI = getgpuinfo

$(GGI): getgpuinfo.cu
	$(NVCC) $(CFLAGS) getgpuinfo.cu -o $(GGI)


clean:
	rm -f $(GGI)

ggi: $(GGI)
	./$(GGI)

