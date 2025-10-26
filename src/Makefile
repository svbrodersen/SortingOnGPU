COMPILER?=nvcc
OPT_FLAGS?=-O3

ELEMS_PER_THREAD=24

MAIN=main

$(MAIN): main.cu host_skel.cuh pbb_kernels.cuh kernels.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(MAIN) $< 

run: main
	./main

# Smoke test target: build and run main, capture output and verify key messages
TEST_LOG?=test.log
.PHONY: test smoke
test: $(MAIN)
	@echo "Running smoke tests (output -> $(TEST_LOG))"
	@./$(MAIN) 2>&1 | tee $(TEST_LOG)
	@echo "Checking expected messages in $(TEST_LOG)"
	@grep -q "Successfully scanInc." $(TEST_LOG) || (echo "FAIL: 'Successfully scanInc.' not found" >&2; exit 2)
	@grep -q "Successfully transpose after scan." $(TEST_LOG) || (echo "FAIL: 'Successfully transpose after scan.' not found" >&2; exit 3)
	@echo "Smoke tests passed."

# Deterministic correctness harness
.PHONY: test-correctness
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

