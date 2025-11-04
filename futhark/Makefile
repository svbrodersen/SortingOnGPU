all: bench

bench:
	futhark bench --backend=cuda --entry-point=sort_u32 --runs=10 \
		Baseline_sort.fut --spec-file=datasets.in; \

clean:
	rm -rf ./Baseline_sort ./Baseline_sort.c ./data
