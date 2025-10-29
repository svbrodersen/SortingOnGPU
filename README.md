# SortingOnGPU

## How to get CUB benchmarking results 
### Complie
```
nvcc -O3 -std=c++14 -lineinfo bench_cub_like_futhark.cu -o bench_cub_like_futhark
```
### Run
```
bash -lc 'for N in 1000 10000 100000 1000000 2000000 5000000 10000000; do ./bench_cub_like_futhark "$N" 10; done' | tee cub_like_futhark.txt

```

#### Results
```
n=1_000:          18μs (95% CI: [       17.5,        18.1])  // 0.056124 Gkeys/s
n=10_000:         102μs (95% CI: [      101.3,       101.8])  // 0.098444 Gkeys/s
n=100_000:         103μs (95% CI: [      102.5,       103.1])  // 0.972672 Gkeys/s
n=1_000_000:         190μs (95% CI: [      189.1,       190.1])  // 5.273016 Gkeys/s
n=2_000_000:         297μs (95% CI: [      294.4,       300.5])  // 6.723322 Gkeys/s
n=5_000_000:         639μs (95% CI: [      638.0,       639.1])  // 7.830039 Gkeys/s
n=10_000_000:        1033μs (95% CI: [     1032.1,      1034.5])  // 9.677559 Gkeys/s
```

## How to get Futhark benchmarking results
### Run
```
futhark bench --backend=cuda --entry-point=sort_u32 --runs=10 Baseline_sort.fut --spec-file=datasets.in --json=futhark_bench.json | tee futhark_bench.txt
```

#### Results
```
Baseline_sort.fut:sort_u32 (no tuning file):
n=1_000:             383μs (95% CI: [     381.8,      383.9])
n=10_000:            391μs (95% CI: [     388.6,      399.5])
n=100_000:           504μs (95% CI: [     503.1,      505.2])
n=1_000_000:        2334μs (95% CI: [    2333.5,     2334.5])
n=2_000_000:        4284μs (95% CI: [    4281.5,     4287.2])
n=5_000_000:        9740μs (95% CI: [    9736.3,     9745.0])
n=10_000_000:      19077μs (95% CI: [   19059.2,    19091.2])
```