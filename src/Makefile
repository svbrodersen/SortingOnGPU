COMPILER?=nvcc
OPT_FLAGS?=-O3 -std=c++17

# Project params (can be overridden on the cmdline)
ELEMS_PER_THREAD ?= 24
BLOCK_SIZE ?= 256
LGH ?= 8
TILES ?= 4096

MAIN := main

MAIN_TEST=radix_test

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $< 

$(MAIN_TEST): radix_test.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN_TEST) $< 

run: $(MAIN)
	./main

test: $(MAIN_TEST)
	./radix_test

# --- Small kernel-specific tests ---
test-k1: test_kernel1
	./test_kernel1

test_kernel1: test_kernel1.cu kernels.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) -o $@ $<

test-k2: test_kernel2
	./test_kernel2

test_kernel2: test_kernel2.cu kernels.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) -o $@ $<

# --- Debug + timing test for kernel2 (one-block; prints µs/M el/s/GB/s) ---
test-both: test_both
	./test_both

test_both: test_both.cu kernels.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) -o $@ $<

# --- Scalable benchmark (optional; times full pass at scale) ---
bench-k2: bench_kernel2
	./bench_kernel2

bench_kernel2: bench_kernel2.cu kernels.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) \
		-DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) \
		-DBLOCK_SIZE=$(BLOCK_SIZE) \
		-DLGH=$(LGH) \
		-DTILES=$(TILES) \
		-o $@ $<

default: compile

clean:
	rm -f $(MAIN)

.PHONY: clean all run

.PHONY: all default compile run clean test-correctness test-k1 test-k2 test-both bench-k2
