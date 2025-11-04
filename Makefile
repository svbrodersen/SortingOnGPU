.DEFAULT_GOAL := all
SUBDIR := src
MAKEFLAGS += --no-print-directory

export COMPILER ?= nvcc
export OPT_FLAGS ?= -O3 -std=c++17 -arch=sm_80
export ELEMS_PER_THREAD ?= 24

.PHONY: all clean validate benchmark cub futhark

all: validate benchmark

validate:
	@echo "===> running validation in $(SUBDIR)/"
	$(MAKE) -C $(SUBDIR) validate

benchmark:
	@echo "===> running benchmark in $(SUBDIR)/"
	$(MAKE) -C $(SUBDIR) benchmark

clean:
	@echo "===> cleaning $(SUBDIR)/"
	$(MAKE) -C $(SUBDIR) clean

cub:
	@echo "===> building CUB benchmark"
	$(MAKE) -C cub-code-radixsort

futhark:
	@echo "===> building Futhark benchmark"
	$(MAKE) -C sorting-on-gpu

plus:
	@echo "===> Our CUDA (validation + benchmark)"
	@bash -lc 'module load cuda >/dev/null 2>&1 || true; \
	  test -x "$$(which nvcc)" || { echo "nvcc not found"; exit 2; }; \
	  $(MAKE) -C src validate benchmark COMPILER=$$(which nvcc)'

	@echo "===> CUB baseline"
	@bash -lc 'module load cuda >/dev/null 2>&1 || true; \
	  test -x "$$(which nvcc)" || { echo "nvcc not found; skipping CUB"; exit 0; }; \
	  if [ -f cub-code-radixsort/bench_cub_like_futhark.cu ]; then \
	    nvcc -O3 -std=c++14 -lineinfo -o cub-code-radixsort/bench_cub_like_futhark cub-code-radixsort/bench_cub_like_futhark.cu && \
	    for N in 1000 10000 100000 1000000 2000000 5000000 10000000; do \
	      cub-code-radixsort/bench_cub_like_futhark $$N 10; \
	    done; \
	  else \
	    echo "CUB sources not found; skipping."; \
	  fi'

	@echo "===> Futhark baseline"
	@bash -lc 'module load futhark >/dev/null 2>&1 || true; \
	  if [ -f src/Baseline_sort.fut ]; then \
	    futhark bench --backend=cuda --entry-point=sort_u32 --runs=10 \
	      src/Baseline_sort.fut --spec-file=src/datasets.in; \
	  else \
	    echo "src/Baseline_sort.fut not found; skipping."; \
	  fi'
