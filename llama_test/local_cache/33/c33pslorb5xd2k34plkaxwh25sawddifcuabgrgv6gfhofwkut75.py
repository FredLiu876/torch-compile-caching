
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x3 = xindex
    x4 = xindex // 128
    x2 = xindex // 1024
    tmp13 = tl.load(in_ptr1 + (x0 + 512*x4), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (128 + x0 + 512*x4), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (256 + x0 + 512*x4), xmask).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (384 + x0 + 512*x4), xmask).to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (x2 + ks0*((x0 % 64))), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x3), xmask & tmp2, other=0.0).to(tl.float32)
    tmp4 = -tmp3
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tmp9 = tmp0 < tmp1
    tmp10 = tl.load(in_ptr0 + (64 + x3), xmask & tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tmp12 = tmp8 + tmp11
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tl_math.cos(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp19 * tmp24
    tmp26 = tmp12 + tmp25
    tl.store(out_ptr0 + (x3), tmp26, xmask)
