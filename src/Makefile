COMPILER?=nvcc
OPT_FLAGS?=-O3

ELEMS_PER_THREAD=24

MAIN=main

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $< 

run: main
	./main

test-correctness: test_correctness
	@echo "Running deterministic correctness harness..."
	./test_correctness

test_correctness: test_correctness.cu constants.cuh
	$(COMPILER) $(OPT_FLAGS) -o test_correctness test_correctness.cu

compile: $(MAIN)

default: compile run

clean:
	rm -f $(MAIN)


.PHONY: clean all run

