all: bench

FUTHARK_FLAGS=--backend=cuda --runs=10

bench:
	futhark bench $(FUTHARK_FLAGS) Baseline_sort.fut --entry-point=sort_u32 --spec-file=dataset_u32.in
	futhark bench $(FUTHARK_FLAGS) Baseline_sort.fut --entry-point=sort_i32 --spec-file=dataset_i32.in
	futhark bench $(FUTHARK_FLAGS) Baseline_sort.fut --entry-point=sort_f32 --spec-file=dataset_f32.in

clean:
	rm -rf ./Baseline_sort ./Baseline_sort.c ./data
