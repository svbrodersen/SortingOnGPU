# SortingOnGPU


## How to get Futhark benchmarking results
```
futhark bench --backend=cuda --entry-point=sort_u32 --runs=10 Baseline_sort.fut --spec-file=datasets.in --json=futhark_bench.json | tee futhark_bench.txt
```
