COMPILER ?= nvcc
OPT_FLAGS ?= -O3 -std=c++17 -arch=sm_80
ELEMS_PER_THREAD ?= 24

MAIN := main
MAIN_TEST := radix_test

.PHONY: all clean validate benchmark

all: validate benchmark

build: $(MAIN) $(MAIN_TEST)

validate: $(MAIN_TEST)
	@echo "Running validation across data types:"
	./$(MAIN_TEST)

bench: $(MAIN)
	for N in 1000 10000 100000 1000000 2000000 5000000 10000000; do \
		./main $$N; \
	done; \

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh constants.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $<

$(MAIN_TEST): radix_test.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN_TEST) $<

clean:
	rm -f $(MAIN) $(MAIN_TEST)

