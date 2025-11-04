-- Baseline_sort.fut
-- Futhark baseline: radix sort for u32 using the sorts package.

module RS = import "lib/github.com/diku-dk/sorts/radix_sort"

entry sort_u32 (xs: []u32) : []u32 =
  -- Generic radix sort needs num_bits + get_bit for the element type.
  RS.radix_sort u32.num_bits u32.get_bit xs