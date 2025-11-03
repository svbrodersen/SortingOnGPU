COMPILER ?= nvcc
OPT_FLAGS ?= -O3 -std=c++17 -arch=sm_80
ELEMS_PER_THREAD ?= 24

MAIN := main
MAIN_TEST := radix_test

all: validate benchmark

validate: $(MAIN_TEST)
	@echo "Running validation across data types:"
	./$(MAIN_TEST)

benchmark: $(MAIN)
	@echo "Running full benchmark inside main.cu:"
	./$(MAIN)

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh constants.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $<

$(MAIN_TEST): radix_test.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN_TEST) $<

clean:
	rm -f $(MAIN) $(MAIN_TEST)

.PHONY: all clean validate benchmark
