# Makefile
NVCC = nvcc

ARCH = -arch=sm_89

CFLAGS = -g -G -O3 -lineinfo $(ARCH) \
         --expt-relaxed-constexpr \
         --extended-lambda \
         --forward-unknown-to-host-compiler \
         -D_FORCE_INLINES \
         -D_MWAITXINTRIN_H_INCLUDED \
         -D__STRICT_ANSI__ \
         -D__USE_MISC \
         -D_GNU_SOURCE \
         -isystem=/usr/local/cuda/include

TARGET = test

$(TARGET): main.cpp solution.cu
	$(NVCC) $(CFLAGS) -include main.cpp solution.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

