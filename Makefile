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
