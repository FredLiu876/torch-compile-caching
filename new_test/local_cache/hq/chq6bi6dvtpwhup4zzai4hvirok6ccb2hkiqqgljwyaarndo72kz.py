
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x3 = xindex
    x2 = xindex // 4096
    tmp27 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x2 + ks0*((x0 % 64))), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x3), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x2 + ks0*((x0 % 64))), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 * tmp8
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tmp0 < tmp1
    tmp16 = tl.load(in_ptr0 + (64 + x3), tmp15, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (x2 + ks0*((x0 % 64))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp18 = tl_math.sin(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp24, tmp13)
    tmp26 = tmp14 + tmp25
    tmp29 = tl_math.cos(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp26 + tmp33
    tl.store(out_ptr0 + (x3), tmp34, None)
