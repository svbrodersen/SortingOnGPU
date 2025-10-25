NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70 -I./

main: main.cu
	$(NVCC) $(NVCCFLAGS) -o main main.cu

run: main
	./main

clean:
	rm -f main
