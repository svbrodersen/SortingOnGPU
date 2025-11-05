# SortingOnGPU

## How to run our CUDA validation and benchmarking
### Compile and Run (from repository root)
### Load modules first
```bash
module load cuda;
module load futhark;
```

Run:
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

## How to run all benchmarks (CUB + Futhark + OURS)
To build and run everything including CUB and Futhark:
```bash
make plus
```

This will:
- Build and run the CUB baseline 
- Run the Futhark baseline
- Build and run our CUDA implementation 

All results are printed to the terminal.

## Cleaning

To clean the build artifacts, simply run: 
```bash
make clean
```

## Notes

Inside ./utils/utils.cuh DEVICE_NUMBER is used to define which device to run
the implementations on. Make sure this value is set to a valid device.
