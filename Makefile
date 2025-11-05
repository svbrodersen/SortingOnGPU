.DEFAULT_GOAL := all
SUBDIR := src
CUBDIR := cub-code-radixsort 
FUTDIR := futhark
MAKEFLAGS += --no-print-directory

COMPILER ?= nvcc
OPT_FLAGS ?= -O3 -std=c++17 -arch=sm_80
ELEMS_PER_THREAD ?= 24

NVCC := nvcc
FUTHARK := futhark


.PHONY: benchmark cub-bench futhark-bench validate clean module_load

all: module_load validate benchmark 

module_load:
	module load cuda;
	module load futhark;

clean:
	$(MAKE) -C $(SUBDIR) clean
	$(MAKE) -C $(CUBDIR) clean
	$(MAKE) -C $(FUTDIR) clean

validate:
	@echo "Running validation in $(SUBDIR)/"
	$(MAKE) -C $(SUBDIR) validate

benchmark:
	@echo "Running benchmark in $(SUBDIR)/"
	$(MAKE) -C $(SUBDIR) bench

cub-bench:
	@echo "Running CUB benchmark"
	$(MAKE) -C $(CUBDIR) bench

futhark-bench:
	@echo "Running Futhark benchmark"
	$(MAKE) -C $(FUTDIR) bench

plus: module_load cub-bench futhark-bench benchmark
