# SortingOnGPU

## How to run our CUDA validation and benchmarking
### Compile and Run (from repository root)
Load modules first:
module load cuda
module load futhark

Then:
make

This will:
- Validate correctness across multiple datatypes and array sizes
- Run the full benchmark from `main.cu` and print timings for several `n`

If you want to run them separately:
make validate
make benchmark

---

## How to run all benchmarks (CUDA + CUB + Futhark)
To build and run everything including CUB and Futhark:
make plus

This will:
- Build and run our CUDA implementation 
- Build and run the CUB baseline 
- Run the Futhark baseline

All results are printed to the terminal.
