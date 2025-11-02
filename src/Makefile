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
	./main 10000000

test: $(MAIN_TEST)
	./radix_test

test_both: test_both.cu kernels.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) -o $@ $<

# ---------- kernel benchmarks ----------
ARCH ?=
RUNS ?= 10
SIZES ?= 1000 10000 100000 1000000 2000000 5000000 10000000 100000000

bench-kernels-build: bench_kernels.cu host_skel.cuh pbb_kernels.cuh kernels.cuh constants.cuh
	$(COMPILER) $(OPT_FLAGS) $(ARCH) bench_kernels.cu -o bench_kernels

bench-k1: bench-kernels-build
	@for N in $(SIZES); do ./bench_kernels --mode k1 --n $$N --runs $(RUNS); done | tee k1.txt

bench-k2: bench-kernels-build
	@for N in $(SIZES); do ./bench_kernels --mode k2 --n $$N --runs $(RUNS); done | tee k2.txt

bench-both: bench-kernels-build
	@for N in $(SIZES); do ./bench_kernels --mode both --n $$N --runs $(RUNS); done | tee k1k2_only.txt

default: compile

clean:
	rm -f $(MAIN) $(MAIN_TEST) bench_kernels k1.txt k2.txt pass_k1k2_only.txt

.PHONY: clean all run

.PHONY: all default compile run clean test-correctness test-k1 test-k2 test-both bench-k2
