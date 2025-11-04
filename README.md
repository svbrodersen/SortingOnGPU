# SortingOnGPU

## How to run our CUDA validation and benchmarking
### Compile and Run (from repository root)
Load modules first
```bash
module load cuda;
module load futhark;
```

Then:
```bash
make
```

This will:
- Validate correctness across multiple datatypes and array sizes
- Run the full benchmark from `main.cu` and print timings for several `n`

If you want to run them separately:
```bash
make validate
make benchmark
```

---

## How to run all benchmarks (CUDA + CUB + Futhark)
To build and run everything including CUB and Futhark:
```bash
make plus
```

This will:
- Build and run our CUDA implementation 
- Build and run the CUB baseline 
- Run the Futhark baseline

All results are printed to the terminal.

## Cleaning

To clean the build artifacts, simply run: 
```bash
make clean
```
