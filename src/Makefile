COMPILER?=nvcc
OPT_FLAGS?=-O3 -std=c++17

ELEMS_PER_THREAD=24

MAIN=main

MAIN_TEST=radix_test

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $< 

$(MAIN_TEST): radix_test.cu host_skel.cuh pbb_kernels.cuh kernels.cuh sort.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN_TEST) $< 

run: $(MAIN)
	./main

test: $(MAIN_TEST)
	./radix_test

compile: $(MAIN)

default: compile run

clean:
	rm -f $(MAIN)

.PHONY: clean all run

