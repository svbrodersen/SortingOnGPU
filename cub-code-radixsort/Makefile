COMPILER ?= nvcc
OPT_FLAGS ?= -O3 -std=c++17 -arch=sm_80 -I$(CUB)/cub 
ELEMS_PER_THREAD ?= 24

all: cub-sort

build: sorting_test.cu helper.cu.h
	$(COMPILER) $(OPT_FLAGS) -o test-cub sorting_test.cu

run: build
	./test-cub 100000000

bench: build
	for N in 1000 10000 100000 1000000 2000000 5000000 10000000; do \
		./test-cub $$N; \
	done; \

clean:
	rm -f test-cub

