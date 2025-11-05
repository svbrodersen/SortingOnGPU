module RS = import "lib/github.com/diku-dk/sorts/radix_sort"

entry sort_u32 (xs: []u32) : []u32 =
  RS.radix_sort u32.num_bits u32.get_bit xs

entry sort_i32 (xs: []i32) : []i32 =
  RS.radix_sort i32.num_bits i32.get_bit xs

entry sort_f32 (xs: []f32) : []f32 =
  RS.radix_sort f32.num_bits f32.get_bit xs
