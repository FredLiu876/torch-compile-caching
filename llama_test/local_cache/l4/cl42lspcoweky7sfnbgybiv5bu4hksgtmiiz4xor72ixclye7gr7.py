# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: ./local_cache/us/cusn5r4ftncl7gqsiixg3hw3ojcfuvaqdhezlxblwecnlckjkjne.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_5, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_98
#   hidden_states => convert_element_type_4
#   hidden_states_1 => mul_93
#   hidden_states_2 => mul_100
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_5 => convert_element_type_5
#   variance => mean
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %primals_2), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %rsqrt), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.float16), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %convert_element_type_5), kwargs = {})
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128256")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 4096*tmp4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp6, r0_mask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp12 = 4096.0
    tmp13 = (tmp10 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp16
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp17 * tmp21
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/3v/c3v53zvz677aerwx3uxwsw5ozzty7wl24rlgzo2n5ut66ygncjhy.py
# Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   position_ids_expanded => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/7z/c7zb5sxn5a2gfktn5y7wudvmg3idkymyzcxbjawowsk6n2pzkgan.py
# Topologically Sorted Source Nodes: [key], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % ks0)
    x3 = xindex // ks1
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + ks0*((x0 % 64))), None, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = x0
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tl.load(in_ptr0 + (64 + 128*x3 + 1024*x1 + (x0)), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tmp7 >= tmp10
    tmp17 = tl.full([1], 128, tl.int64)
    tmp18 = tmp7 < tmp17
    tmp19 = tl.load(in_ptr0 + (128*x3 + 1024*x1 + ((-64) + x0)), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp11, tmp15, tmp19)
    tmp21 = tl_math.sin(tmp1)
    tmp22 = tmp21 * tmp3
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp6 + tmp24
    tl.store(out_ptr0 + (x5), tmp25, None)
''', device_str='cuda')


# kernel path: ./local_cache/gb/cgbztvfhheweq52nxjalsu4azzntobt3zuoh4y4ngzxoiat5z4gn.py
# Topologically Sorted Source Nodes: [value], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   value => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./local_cache/74/c74cu5piannu2kyts54gyevikcp25xbtpxgzzz6hmumgpkx3ahdw.py
# Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
# Source node to ATen node mapping:
#   cat_1 => cat
#   mul_4 => mul_168
#   mul_5 => mul_185
#   q_embed => add_191
#   query => clone_4
# Graph fragment:
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_5), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_6), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_168, %mul_185), kwargs = {})
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_191,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_cat_clone_mul_4 = async_compile.triton('triton_poi_fused_add_cat_clone_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_clone_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_clone_mul_4(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 4096
    x4 = xindex // 128
    x1 = ((xindex // 128) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + ks0*((x0 % 64))), None, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = x0
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tl.load(in_ptr0 + (64 + 128*x4 + (x0)), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tmp7 >= tmp10
    tmp17 = tl.full([1], 128, tl.int64)
    tmp18 = tmp7 < tmp17
    tmp19 = tl.load(in_ptr0 + (128*x4 + ((-64) + x0)), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp11, tmp15, tmp19)
    tmp21 = tl_math.sin(tmp1)
    tmp22 = tmp21 * tmp3
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp6 + tmp24
    tl.store(out_ptr0 + (x0 + 128*x2 + 128*ks0*x1), tmp25, None)
''', device_str='cuda')


# kernel path: ./local_cache/o7/co7iw7dsuutkvnklatkhtnweaaonzxd2sjk2cozzthzuyfwph2qu.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.scalar_tensor, aten.where, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   attn_output => constant_pad_nd, full_default_1, full_default_2, where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_108], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_scalar_tensor_where_5 = async_compile.triton('triton_poi_fused_constant_pad_nd_scalar_tensor_where_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_scalar_tensor_where_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_scalar_tensor_where_5(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = x0
    tmp1 = ks1
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = x1
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], True, tl.int1)
    tmp7 = tmp6 & tmp5
    tl.device_assert((x0 < ks1) | ~(xmask & tmp2), "index out of bounds: x0 < ks1")
    tmp9 = tl.load(in_ptr0 + (x0), xmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp10 = (tmp9 != 0)
    tmp11 = tmp7 & tmp10
    tmp12 = 0.0
    tmp13 = float("-inf")
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/xb/cxbiz6a53yuzne7tuzqghtuipdavwtl2jp3relur3ksonmtcg44q.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, pow_2, variance_1, add_8, rsqrt_1, hidden_states_7, to_7, hidden_states_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_8 => add_336
#   hidden_states_5 => add_323
#   hidden_states_6 => convert_element_type_14
#   hidden_states_7 => mul_402
#   hidden_states_8 => mul_409
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_7 => convert_element_type_15
#   variance_1 => mean_1
# Graph fragment:
#   %add_323 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_323, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_336,), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_402, torch.float16), kwargs = {})
#   %mul_409 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, %convert_element_type_15), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = 4096.0
    tmp9 = (tmp6 / tmp8)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 * tmp12
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp13 * tmp19
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/zl/czlnnli5wdnzsq3xf3gaijpvc6mijjwympjjhftw2jsmqxb4oqpa.py
# Topologically Sorted Source Nodes: [silu, mul_10], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_10 => mul_443
#   silu => convert_element_type_18, convert_element_type_19, mul_427, sigmoid
# Graph fragment:
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_24, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_427, torch.float16), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %view_26), kwargs = {})
triton_poi_fused_mul_silu_7 = async_compile.triton('triton_poi_fused_mul_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/ir/cirveghol7wjrq7zwkkgu3sphruzjhq4ru5zcvmudgaexdo5niag.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_9, hidden_states_10, pow_3, variance_2, add_10, rsqrt_2, hidden_states_11, to_9, hidden_states_12], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_401
#   hidden_states_10 => convert_element_type_24
#   hidden_states_11 => mul_479
#   hidden_states_12 => mul_486
#   hidden_states_5 => add_323
#   hidden_states_9 => add_388
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_9 => convert_element_type_25
#   variance_2 => mean_2
# Graph fragment:
#   %add_323 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %add_388 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_323, %view_28), kwargs = {})
#   %convert_element_type_24 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_388, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_24, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_401 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_401,), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %rsqrt_2), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_479, torch.float16), kwargs = {})
#   %mul_486 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_15, %convert_element_type_25), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp4, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = 4096.0
    tmp11 = (tmp8 / tmp10)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp14, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 * tmp14
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp15 * tmp19
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/cp/ccpmvf6lgmnpforavvbrvwi4t5zoiosfy62jj2u37b4g5lsm5yib.py
# Topologically Sorted Source Nodes: [hidden_states_15, hidden_states_16, pow_4, variance_3, add_14, rsqrt_3, hidden_states_17, to_11, hidden_states_18], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_14 => add_639
#   hidden_states_15 => add_626
#   hidden_states_16 => convert_element_type_34
#   hidden_states_17 => mul_788
#   hidden_states_18 => mul_795
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_11 => convert_element_type_35
#   variance_3 => mean_3
# Graph fragment:
#   %add_626 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_388, %view_42), kwargs = {})
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_626, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_34, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_639 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_639,), kwargs = {})
#   %mul_788 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_34, %rsqrt_3), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_788, torch.float16), kwargs = {})
#   %mul_795 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_20, %convert_element_type_35), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp2, r0_mask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = 4096.0
    tmp9 = (tmp6 / tmp8)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp12, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp12
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp13 * tmp17
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295 = args
    args.clear()
    s0 = primals_1
    assert_size_stride(primals_2, (1, s0), (s0, 1))
    assert_size_stride(primals_3, (128256, 4096), (4096, 1))
    assert_size_stride(primals_4, (1, s0), (s0, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (4096, ), (1, ))
    assert_size_stride(primals_7, (4096, 4096), (4096, 1))
    assert_size_stride(primals_8, (1024, 4096), (4096, 1))
    assert_size_stride(primals_9, (1024, 4096), (4096, 1))
    assert_size_stride(primals_10, (4096, 4096), (4096, 1))
    assert_size_stride(primals_11, (4096, ), (1, ))
    assert_size_stride(primals_12, (14336, 4096), (4096, 1))
    assert_size_stride(primals_13, (14336, 4096), (4096, 1))
    assert_size_stride(primals_14, (4096, 14336), (14336, 1))
    assert_size_stride(primals_15, (4096, ), (1, ))
    assert_size_stride(primals_16, (4096, 4096), (4096, 1))
    assert_size_stride(primals_17, (1024, 4096), (4096, 1))
    assert_size_stride(primals_18, (1024, 4096), (4096, 1))
    assert_size_stride(primals_19, (4096, 4096), (4096, 1))
    assert_size_stride(primals_20, (4096, ), (1, ))
    assert_size_stride(primals_21, (14336, 4096), (4096, 1))
    assert_size_stride(primals_22, (14336, 4096), (4096, 1))
    assert_size_stride(primals_23, (4096, 14336), (14336, 1))
    assert_size_stride(primals_24, (4096, ), (1, ))
    assert_size_stride(primals_25, (4096, 4096), (4096, 1))
    assert_size_stride(primals_26, (1024, 4096), (4096, 1))
    assert_size_stride(primals_27, (1024, 4096), (4096, 1))
    assert_size_stride(primals_28, (4096, 4096), (4096, 1))
    assert_size_stride(primals_29, (4096, ), (1, ))
    assert_size_stride(primals_30, (14336, 4096), (4096, 1))
    assert_size_stride(primals_31, (14336, 4096), (4096, 1))
    assert_size_stride(primals_32, (4096, 14336), (14336, 1))
    assert_size_stride(primals_33, (4096, ), (1, ))
    assert_size_stride(primals_34, (4096, 4096), (4096, 1))
    assert_size_stride(primals_35, (1024, 4096), (4096, 1))
    assert_size_stride(primals_36, (1024, 4096), (4096, 1))
    assert_size_stride(primals_37, (4096, 4096), (4096, 1))
    assert_size_stride(primals_38, (4096, ), (1, ))
    assert_size_stride(primals_39, (14336, 4096), (4096, 1))
    assert_size_stride(primals_40, (14336, 4096), (4096, 1))
    assert_size_stride(primals_41, (4096, 14336), (14336, 1))
    assert_size_stride(primals_42, (4096, ), (1, ))
    assert_size_stride(primals_43, (4096, 4096), (4096, 1))
    assert_size_stride(primals_44, (1024, 4096), (4096, 1))
    assert_size_stride(primals_45, (1024, 4096), (4096, 1))
    assert_size_stride(primals_46, (4096, 4096), (4096, 1))
    assert_size_stride(primals_47, (4096, ), (1, ))
    assert_size_stride(primals_48, (14336, 4096), (4096, 1))
    assert_size_stride(primals_49, (14336, 4096), (4096, 1))
    assert_size_stride(primals_50, (4096, 14336), (14336, 1))
    assert_size_stride(primals_51, (4096, ), (1, ))
    assert_size_stride(primals_52, (4096, 4096), (4096, 1))
    assert_size_stride(primals_53, (1024, 4096), (4096, 1))
    assert_size_stride(primals_54, (1024, 4096), (4096, 1))
    assert_size_stride(primals_55, (4096, 4096), (4096, 1))
    assert_size_stride(primals_56, (4096, ), (1, ))
    assert_size_stride(primals_57, (14336, 4096), (4096, 1))
    assert_size_stride(primals_58, (14336, 4096), (4096, 1))
    assert_size_stride(primals_59, (4096, 14336), (14336, 1))
    assert_size_stride(primals_60, (4096, ), (1, ))
    assert_size_stride(primals_61, (4096, 4096), (4096, 1))
    assert_size_stride(primals_62, (1024, 4096), (4096, 1))
    assert_size_stride(primals_63, (1024, 4096), (4096, 1))
    assert_size_stride(primals_64, (4096, 4096), (4096, 1))
    assert_size_stride(primals_65, (4096, ), (1, ))
    assert_size_stride(primals_66, (14336, 4096), (4096, 1))
    assert_size_stride(primals_67, (14336, 4096), (4096, 1))
    assert_size_stride(primals_68, (4096, 14336), (14336, 1))
    assert_size_stride(primals_69, (4096, ), (1, ))
    assert_size_stride(primals_70, (4096, 4096), (4096, 1))
    assert_size_stride(primals_71, (1024, 4096), (4096, 1))
    assert_size_stride(primals_72, (1024, 4096), (4096, 1))
    assert_size_stride(primals_73, (4096, 4096), (4096, 1))
    assert_size_stride(primals_74, (4096, ), (1, ))
    assert_size_stride(primals_75, (14336, 4096), (4096, 1))
    assert_size_stride(primals_76, (14336, 4096), (4096, 1))
    assert_size_stride(primals_77, (4096, 14336), (14336, 1))
    assert_size_stride(primals_78, (4096, ), (1, ))
    assert_size_stride(primals_79, (4096, 4096), (4096, 1))
    assert_size_stride(primals_80, (1024, 4096), (4096, 1))
    assert_size_stride(primals_81, (1024, 4096), (4096, 1))
    assert_size_stride(primals_82, (4096, 4096), (4096, 1))
    assert_size_stride(primals_83, (4096, ), (1, ))
    assert_size_stride(primals_84, (14336, 4096), (4096, 1))
    assert_size_stride(primals_85, (14336, 4096), (4096, 1))
    assert_size_stride(primals_86, (4096, 14336), (14336, 1))
    assert_size_stride(primals_87, (4096, ), (1, ))
    assert_size_stride(primals_88, (4096, 4096), (4096, 1))
    assert_size_stride(primals_89, (1024, 4096), (4096, 1))
    assert_size_stride(primals_90, (1024, 4096), (4096, 1))
    assert_size_stride(primals_91, (4096, 4096), (4096, 1))
    assert_size_stride(primals_92, (4096, ), (1, ))
    assert_size_stride(primals_93, (14336, 4096), (4096, 1))
    assert_size_stride(primals_94, (14336, 4096), (4096, 1))
    assert_size_stride(primals_95, (4096, 14336), (14336, 1))
    assert_size_stride(primals_96, (4096, ), (1, ))
    assert_size_stride(primals_97, (4096, 4096), (4096, 1))
    assert_size_stride(primals_98, (1024, 4096), (4096, 1))
    assert_size_stride(primals_99, (1024, 4096), (4096, 1))
    assert_size_stride(primals_100, (4096, 4096), (4096, 1))
    assert_size_stride(primals_101, (4096, ), (1, ))
    assert_size_stride(primals_102, (14336, 4096), (4096, 1))
    assert_size_stride(primals_103, (14336, 4096), (4096, 1))
    assert_size_stride(primals_104, (4096, 14336), (14336, 1))
    assert_size_stride(primals_105, (4096, ), (1, ))
    assert_size_stride(primals_106, (4096, 4096), (4096, 1))
    assert_size_stride(primals_107, (1024, 4096), (4096, 1))
    assert_size_stride(primals_108, (1024, 4096), (4096, 1))
    assert_size_stride(primals_109, (4096, 4096), (4096, 1))
    assert_size_stride(primals_110, (4096, ), (1, ))
    assert_size_stride(primals_111, (14336, 4096), (4096, 1))
    assert_size_stride(primals_112, (14336, 4096), (4096, 1))
    assert_size_stride(primals_113, (4096, 14336), (14336, 1))
    assert_size_stride(primals_114, (4096, ), (1, ))
    assert_size_stride(primals_115, (4096, 4096), (4096, 1))
    assert_size_stride(primals_116, (1024, 4096), (4096, 1))
    assert_size_stride(primals_117, (1024, 4096), (4096, 1))
    assert_size_stride(primals_118, (4096, 4096), (4096, 1))
    assert_size_stride(primals_119, (4096, ), (1, ))
    assert_size_stride(primals_120, (14336, 4096), (4096, 1))
    assert_size_stride(primals_121, (14336, 4096), (4096, 1))
    assert_size_stride(primals_122, (4096, 14336), (14336, 1))
    assert_size_stride(primals_123, (4096, ), (1, ))
    assert_size_stride(primals_124, (4096, 4096), (4096, 1))
    assert_size_stride(primals_125, (1024, 4096), (4096, 1))
    assert_size_stride(primals_126, (1024, 4096), (4096, 1))
    assert_size_stride(primals_127, (4096, 4096), (4096, 1))
    assert_size_stride(primals_128, (4096, ), (1, ))
    assert_size_stride(primals_129, (14336, 4096), (4096, 1))
    assert_size_stride(primals_130, (14336, 4096), (4096, 1))
    assert_size_stride(primals_131, (4096, 14336), (14336, 1))
    assert_size_stride(primals_132, (4096, ), (1, ))
    assert_size_stride(primals_133, (4096, 4096), (4096, 1))
    assert_size_stride(primals_134, (1024, 4096), (4096, 1))
    assert_size_stride(primals_135, (1024, 4096), (4096, 1))
    assert_size_stride(primals_136, (4096, 4096), (4096, 1))
    assert_size_stride(primals_137, (4096, ), (1, ))
    assert_size_stride(primals_138, (14336, 4096), (4096, 1))
    assert_size_stride(primals_139, (14336, 4096), (4096, 1))
    assert_size_stride(primals_140, (4096, 14336), (14336, 1))
    assert_size_stride(primals_141, (4096, ), (1, ))
    assert_size_stride(primals_142, (4096, 4096), (4096, 1))
    assert_size_stride(primals_143, (1024, 4096), (4096, 1))
    assert_size_stride(primals_144, (1024, 4096), (4096, 1))
    assert_size_stride(primals_145, (4096, 4096), (4096, 1))
    assert_size_stride(primals_146, (4096, ), (1, ))
    assert_size_stride(primals_147, (14336, 4096), (4096, 1))
    assert_size_stride(primals_148, (14336, 4096), (4096, 1))
    assert_size_stride(primals_149, (4096, 14336), (14336, 1))
    assert_size_stride(primals_150, (4096, ), (1, ))
    assert_size_stride(primals_151, (4096, 4096), (4096, 1))
    assert_size_stride(primals_152, (1024, 4096), (4096, 1))
    assert_size_stride(primals_153, (1024, 4096), (4096, 1))
    assert_size_stride(primals_154, (4096, 4096), (4096, 1))
    assert_size_stride(primals_155, (4096, ), (1, ))
    assert_size_stride(primals_156, (14336, 4096), (4096, 1))
    assert_size_stride(primals_157, (14336, 4096), (4096, 1))
    assert_size_stride(primals_158, (4096, 14336), (14336, 1))
    assert_size_stride(primals_159, (4096, ), (1, ))
    assert_size_stride(primals_160, (4096, 4096), (4096, 1))
    assert_size_stride(primals_161, (1024, 4096), (4096, 1))
    assert_size_stride(primals_162, (1024, 4096), (4096, 1))
    assert_size_stride(primals_163, (4096, 4096), (4096, 1))
    assert_size_stride(primals_164, (4096, ), (1, ))
    assert_size_stride(primals_165, (14336, 4096), (4096, 1))
    assert_size_stride(primals_166, (14336, 4096), (4096, 1))
    assert_size_stride(primals_167, (4096, 14336), (14336, 1))
    assert_size_stride(primals_168, (4096, ), (1, ))
    assert_size_stride(primals_169, (4096, 4096), (4096, 1))
    assert_size_stride(primals_170, (1024, 4096), (4096, 1))
    assert_size_stride(primals_171, (1024, 4096), (4096, 1))
    assert_size_stride(primals_172, (4096, 4096), (4096, 1))
    assert_size_stride(primals_173, (4096, ), (1, ))
    assert_size_stride(primals_174, (14336, 4096), (4096, 1))
    assert_size_stride(primals_175, (14336, 4096), (4096, 1))
    assert_size_stride(primals_176, (4096, 14336), (14336, 1))
    assert_size_stride(primals_177, (4096, ), (1, ))
    assert_size_stride(primals_178, (4096, 4096), (4096, 1))
    assert_size_stride(primals_179, (1024, 4096), (4096, 1))
    assert_size_stride(primals_180, (1024, 4096), (4096, 1))
    assert_size_stride(primals_181, (4096, 4096), (4096, 1))
    assert_size_stride(primals_182, (4096, ), (1, ))
    assert_size_stride(primals_183, (14336, 4096), (4096, 1))
    assert_size_stride(primals_184, (14336, 4096), (4096, 1))
    assert_size_stride(primals_185, (4096, 14336), (14336, 1))
    assert_size_stride(primals_186, (4096, ), (1, ))
    assert_size_stride(primals_187, (4096, 4096), (4096, 1))
    assert_size_stride(primals_188, (1024, 4096), (4096, 1))
    assert_size_stride(primals_189, (1024, 4096), (4096, 1))
    assert_size_stride(primals_190, (4096, 4096), (4096, 1))
    assert_size_stride(primals_191, (4096, ), (1, ))
    assert_size_stride(primals_192, (14336, 4096), (4096, 1))
    assert_size_stride(primals_193, (14336, 4096), (4096, 1))
    assert_size_stride(primals_194, (4096, 14336), (14336, 1))
    assert_size_stride(primals_195, (4096, ), (1, ))
    assert_size_stride(primals_196, (4096, 4096), (4096, 1))
    assert_size_stride(primals_197, (1024, 4096), (4096, 1))
    assert_size_stride(primals_198, (1024, 4096), (4096, 1))
    assert_size_stride(primals_199, (4096, 4096), (4096, 1))
    assert_size_stride(primals_200, (4096, ), (1, ))
    assert_size_stride(primals_201, (14336, 4096), (4096, 1))
    assert_size_stride(primals_202, (14336, 4096), (4096, 1))
    assert_size_stride(primals_203, (4096, 14336), (14336, 1))
    assert_size_stride(primals_204, (4096, ), (1, ))
    assert_size_stride(primals_205, (4096, 4096), (4096, 1))
    assert_size_stride(primals_206, (1024, 4096), (4096, 1))
    assert_size_stride(primals_207, (1024, 4096), (4096, 1))
    assert_size_stride(primals_208, (4096, 4096), (4096, 1))
    assert_size_stride(primals_209, (4096, ), (1, ))
    assert_size_stride(primals_210, (14336, 4096), (4096, 1))
    assert_size_stride(primals_211, (14336, 4096), (4096, 1))
    assert_size_stride(primals_212, (4096, 14336), (14336, 1))
    assert_size_stride(primals_213, (4096, ), (1, ))
    assert_size_stride(primals_214, (4096, 4096), (4096, 1))
    assert_size_stride(primals_215, (1024, 4096), (4096, 1))
    assert_size_stride(primals_216, (1024, 4096), (4096, 1))
    assert_size_stride(primals_217, (4096, 4096), (4096, 1))
    assert_size_stride(primals_218, (4096, ), (1, ))
    assert_size_stride(primals_219, (14336, 4096), (4096, 1))
    assert_size_stride(primals_220, (14336, 4096), (4096, 1))
    assert_size_stride(primals_221, (4096, 14336), (14336, 1))
    assert_size_stride(primals_222, (4096, ), (1, ))
    assert_size_stride(primals_223, (4096, 4096), (4096, 1))
    assert_size_stride(primals_224, (1024, 4096), (4096, 1))
    assert_size_stride(primals_225, (1024, 4096), (4096, 1))
    assert_size_stride(primals_226, (4096, 4096), (4096, 1))
    assert_size_stride(primals_227, (4096, ), (1, ))
    assert_size_stride(primals_228, (14336, 4096), (4096, 1))
    assert_size_stride(primals_229, (14336, 4096), (4096, 1))
    assert_size_stride(primals_230, (4096, 14336), (14336, 1))
    assert_size_stride(primals_231, (4096, ), (1, ))
    assert_size_stride(primals_232, (4096, 4096), (4096, 1))
    assert_size_stride(primals_233, (1024, 4096), (4096, 1))
    assert_size_stride(primals_234, (1024, 4096), (4096, 1))
    assert_size_stride(primals_235, (4096, 4096), (4096, 1))
    assert_size_stride(primals_236, (4096, ), (1, ))
    assert_size_stride(primals_237, (14336, 4096), (4096, 1))
    assert_size_stride(primals_238, (14336, 4096), (4096, 1))
    assert_size_stride(primals_239, (4096, 14336), (14336, 1))
    assert_size_stride(primals_240, (4096, ), (1, ))
    assert_size_stride(primals_241, (4096, 4096), (4096, 1))
    assert_size_stride(primals_242, (1024, 4096), (4096, 1))
    assert_size_stride(primals_243, (1024, 4096), (4096, 1))
    assert_size_stride(primals_244, (4096, 4096), (4096, 1))
    assert_size_stride(primals_245, (4096, ), (1, ))
    assert_size_stride(primals_246, (14336, 4096), (4096, 1))
    assert_size_stride(primals_247, (14336, 4096), (4096, 1))
    assert_size_stride(primals_248, (4096, 14336), (14336, 1))
    assert_size_stride(primals_249, (4096, ), (1, ))
    assert_size_stride(primals_250, (4096, 4096), (4096, 1))
    assert_size_stride(primals_251, (1024, 4096), (4096, 1))
    assert_size_stride(primals_252, (1024, 4096), (4096, 1))
    assert_size_stride(primals_253, (4096, 4096), (4096, 1))
    assert_size_stride(primals_254, (4096, ), (1, ))
    assert_size_stride(primals_255, (14336, 4096), (4096, 1))
    assert_size_stride(primals_256, (14336, 4096), (4096, 1))
    assert_size_stride(primals_257, (4096, 14336), (14336, 1))
    assert_size_stride(primals_258, (4096, ), (1, ))
    assert_size_stride(primals_259, (4096, 4096), (4096, 1))
    assert_size_stride(primals_260, (1024, 4096), (4096, 1))
    assert_size_stride(primals_261, (1024, 4096), (4096, 1))
    assert_size_stride(primals_262, (4096, 4096), (4096, 1))
    assert_size_stride(primals_263, (4096, ), (1, ))
    assert_size_stride(primals_264, (14336, 4096), (4096, 1))
    assert_size_stride(primals_265, (14336, 4096), (4096, 1))
    assert_size_stride(primals_266, (4096, 14336), (14336, 1))
    assert_size_stride(primals_267, (4096, ), (1, ))
    assert_size_stride(primals_268, (4096, 4096), (4096, 1))
    assert_size_stride(primals_269, (1024, 4096), (4096, 1))
    assert_size_stride(primals_270, (1024, 4096), (4096, 1))
    assert_size_stride(primals_271, (4096, 4096), (4096, 1))
    assert_size_stride(primals_272, (4096, ), (1, ))
    assert_size_stride(primals_273, (14336, 4096), (4096, 1))
    assert_size_stride(primals_274, (14336, 4096), (4096, 1))
    assert_size_stride(primals_275, (4096, 14336), (14336, 1))
    assert_size_stride(primals_276, (4096, ), (1, ))
    assert_size_stride(primals_277, (4096, 4096), (4096, 1))
    assert_size_stride(primals_278, (1024, 4096), (4096, 1))
    assert_size_stride(primals_279, (1024, 4096), (4096, 1))
    assert_size_stride(primals_280, (4096, 4096), (4096, 1))
    assert_size_stride(primals_281, (4096, ), (1, ))
    assert_size_stride(primals_282, (14336, 4096), (4096, 1))
    assert_size_stride(primals_283, (14336, 4096), (4096, 1))
    assert_size_stride(primals_284, (4096, 14336), (14336, 1))
    assert_size_stride(primals_285, (4096, ), (1, ))
    assert_size_stride(primals_286, (4096, 4096), (4096, 1))
    assert_size_stride(primals_287, (1024, 4096), (4096, 1))
    assert_size_stride(primals_288, (1024, 4096), (4096, 1))
    assert_size_stride(primals_289, (4096, 4096), (4096, 1))
    assert_size_stride(primals_290, (4096, ), (1, ))
    assert_size_stride(primals_291, (14336, 4096), (4096, 1))
    assert_size_stride(primals_292, (14336, 4096), (4096, 1))
    assert_size_stride(primals_293, (4096, 14336), (14336, 1))
    assert_size_stride(primals_294, (4096, ), (1, ))
    assert_size_stride(primals_295, (128256, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        buf3 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf4 = reinterpret_tensor(buf3, (1, s0, 1), (s0, 1, 1), 0); del buf3  # reuse
        buf5 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_5, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(buf4, primals_2, primals_3, primals_6, buf0, buf5, s0, 4096, stream=stream0)
        del primals_3
        buf1 = empty_strided_cuda((1, 1, s0), (s0, s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf1, s0, stream=stream0)
        buf2 = empty_strided_cuda((1, 64, s0), (64*s0, s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded, matmul], Original ATen: [aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_5, (1, 64, 1), (0, 1, 0), 0), buf1, out=buf2)
        del primals_5
        buf6 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_7, (4096, 4096), (1, 4096), 0), out=buf6)
        buf7 = empty_strided_cuda((s0, 1024), (1024, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_8, (4096, 1024), (1, 4096), 0), out=buf7)
        buf8 = empty_strided_cuda((s0, 1024), (1024, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_9, (4096, 1024), (1, 4096), 0), out=buf8)
        ps0 = 512*s0
        buf9 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf7, buf2, buf9, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf10 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf8, buf10, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf11 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf6, buf2, buf11, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        ps1 = 8 + s0 + (-1)*(s0 % 8)
        buf12 = empty_strided_cuda((1, 1, s0, 8 + s0 + (-1)*(s0 % 8)), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 1, 8 + s0 + (-1)*(s0 % 8), 1), torch.float16)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.scalar_tensor, aten.where, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_scalar_tensor_where_5_xnumel = s0*s0 + 8*s0 + (-1)*s0*(s0 % 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_scalar_tensor_where_5.run(primals_4, buf12, ps1, s0, triton_poi_fused_constant_pad_nd_scalar_tensor_where_5_xnumel, stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf11, reinterpret_tensor(buf9, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf10, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf14 = buf13[0]
        assert_size_stride(buf14, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf15 = buf13[1]
        assert_size_stride(buf15, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf16 = buf13[2]
        assert_size_stride(buf16, (), ())
        buf17 = buf13[3]
        assert_size_stride(buf17, (), ())
        del buf13
        buf18 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_10, (4096, 4096), (1, 4096), 0), out=buf18)
        buf19 = reinterpret_tensor(buf1, (1, s0, 1), (s0, 1, s0), 0); del buf1  # reuse
        buf20 = reinterpret_tensor(buf19, (1, s0, 1), (s0, 1, 1), 0); del buf19  # reuse
        buf21 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, pow_2, variance_1, add_8, rsqrt_1, hidden_states_7, to_7, hidden_states_8], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf20, buf0, buf18, primals_11, buf21, s0, 4096, stream=stream0)
        buf22 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_12, (4096, 14336), (1, 4096), 0), out=buf22)
        buf23 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_13, (4096, 14336), (1, 4096), 0), out=buf23)
        buf24 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu, mul_10], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf22, buf23, buf24, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf25 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_14, (14336, 4096), (1, 14336), 0), out=buf25)
        buf26 = reinterpret_tensor(buf25, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf28 = reinterpret_tensor(buf27, (1, s0, 1), (s0, 1, 1), 0); del buf27  # reuse
        buf29 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_9, hidden_states_10, pow_3, variance_2, add_10, rsqrt_2, hidden_states_11, to_9, hidden_states_12], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf26, buf28, buf0, buf18, primals_15, buf29, s0, 4096, stream=stream0)
        buf30 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_16, (4096, 4096), (1, 4096), 0), out=buf30)
        buf31 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_17, (4096, 1024), (1, 4096), 0), out=buf31)
        buf32 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_18, (4096, 1024), (1, 4096), 0), out=buf32)
        buf33 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf31, buf2, buf33, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf34 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf32, buf34, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf35 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_13, cat_3, mul_14, q_embed_1, query_1], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf30, buf2, buf35, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf36 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf35, reinterpret_tensor(buf33, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf34, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf37 = buf36[0]
        assert_size_stride(buf37, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf38 = buf36[1]
        assert_size_stride(buf38, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf39 = buf36[2]
        assert_size_stride(buf39, (), ())
        buf40 = buf36[3]
        assert_size_stride(buf40, (), ())
        del buf36
        buf41 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_19, (4096, 4096), (1, 4096), 0), out=buf41)
        buf42 = reinterpret_tensor(buf41, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf41  # reuse
        buf43 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf44 = reinterpret_tensor(buf43, (1, s0, 1), (s0, 1, 1), 0); del buf43  # reuse
        buf45 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_15, hidden_states_16, pow_4, variance_3, add_14, rsqrt_3, hidden_states_17, to_11, hidden_states_18], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf42, buf44, buf26, primals_20, buf45, s0, 4096, stream=stream0)
        buf46 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_21, (4096, 14336), (1, 4096), 0), out=buf46)
        buf47 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_22, (4096, 14336), (1, 4096), 0), out=buf47)
        buf48 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_1, mul_19], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf46, buf47, buf48, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf49 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_23, (14336, 4096), (1, 14336), 0), out=buf49)
        buf50 = reinterpret_tensor(buf49, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf49  # reuse
        buf51 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf52 = reinterpret_tensor(buf51, (1, s0, 1), (s0, 1, 1), 0); del buf51  # reuse
        buf53 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, pow_5, variance_4, add_16, rsqrt_4, hidden_states_21, to_13, hidden_states_22], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf50, buf52, buf42, primals_24, buf53, s0, 4096, stream=stream0)
        buf54 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_25, (4096, 4096), (1, 4096), 0), out=buf54)
        buf55 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_26, (4096, 1024), (1, 4096), 0), out=buf55)
        buf56 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_27, (4096, 1024), (1, 4096), 0), out=buf56)
        buf57 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf55, buf2, buf57, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf58 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf56, buf58, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf59 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_22, cat_5, mul_23, q_embed_2, query_2], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf54, buf2, buf59, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf59, reinterpret_tensor(buf57, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf58, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf61 = buf60[0]
        assert_size_stride(buf61, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf62 = buf60[1]
        assert_size_stride(buf62, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf63 = buf60[2]
        assert_size_stride(buf63, (), ())
        buf64 = buf60[3]
        assert_size_stride(buf64, (), ())
        del buf60
        buf65 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_28, (4096, 4096), (1, 4096), 0), out=buf65)
        buf66 = reinterpret_tensor(buf65, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf65  # reuse
        buf67 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf68 = reinterpret_tensor(buf67, (1, s0, 1), (s0, 1, 1), 0); del buf67  # reuse
        buf69 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, pow_6, variance_5, add_20, rsqrt_5, hidden_states_27, to_15, hidden_states_28], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf66, buf68, buf50, primals_29, buf69, s0, 4096, stream=stream0)
        buf70 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_30, (4096, 14336), (1, 4096), 0), out=buf70)
        buf71 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_31, (4096, 14336), (1, 4096), 0), out=buf71)
        buf72 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_2, mul_28], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf70, buf71, buf72, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf73 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_32, (14336, 4096), (1, 14336), 0), out=buf73)
        buf74 = reinterpret_tensor(buf73, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf73  # reuse
        buf75 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf76 = reinterpret_tensor(buf75, (1, s0, 1), (s0, 1, 1), 0); del buf75  # reuse
        buf77 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_29, hidden_states_30, pow_7, variance_6, add_22, rsqrt_6, hidden_states_31, to_17, hidden_states_32], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf74, buf76, buf66, primals_33, buf77, s0, 4096, stream=stream0)
        buf78 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_34, (4096, 4096), (1, 4096), 0), out=buf78)
        buf79 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_35, (4096, 1024), (1, 4096), 0), out=buf79)
        buf80 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_36, (4096, 1024), (1, 4096), 0), out=buf80)
        buf81 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf79, buf2, buf81, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf82 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf80, buf82, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf83 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_31, cat_7, mul_32, q_embed_3, query_3], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf78, buf2, buf83, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf84 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf83, reinterpret_tensor(buf81, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf82, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf85 = buf84[0]
        assert_size_stride(buf85, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf86 = buf84[1]
        assert_size_stride(buf86, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf87 = buf84[2]
        assert_size_stride(buf87, (), ())
        buf88 = buf84[3]
        assert_size_stride(buf88, (), ())
        del buf84
        buf89 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_37, (4096, 4096), (1, 4096), 0), out=buf89)
        buf90 = reinterpret_tensor(buf89, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf89  # reuse
        buf91 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf92 = reinterpret_tensor(buf91, (1, s0, 1), (s0, 1, 1), 0); del buf91  # reuse
        buf93 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_19, hidden_states_38], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf90, buf92, buf74, primals_38, buf93, s0, 4096, stream=stream0)
        buf94 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_39, (4096, 14336), (1, 4096), 0), out=buf94)
        buf95 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_40, (4096, 14336), (1, 4096), 0), out=buf95)
        buf96 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_3, mul_37], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf94, buf95, buf96, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf97 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_41, (14336, 4096), (1, 14336), 0), out=buf97)
        buf98 = reinterpret_tensor(buf97, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf97  # reuse
        buf99 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf100 = reinterpret_tensor(buf99, (1, s0, 1), (s0, 1, 1), 0); del buf99  # reuse
        buf101 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_21, hidden_states_42], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf98, buf100, buf90, primals_42, buf101, s0, 4096, stream=stream0)
        buf102 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_43, (4096, 4096), (1, 4096), 0), out=buf102)
        buf103 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_44, (4096, 1024), (1, 4096), 0), out=buf103)
        buf104 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_45, (4096, 1024), (1, 4096), 0), out=buf104)
        buf105 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf103, buf2, buf105, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf106 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf104, buf106, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf107 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_40, cat_9, mul_41, q_embed_4, query_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf102, buf2, buf107, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_16], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf108 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf107, reinterpret_tensor(buf105, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf106, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf109 = buf108[0]
        assert_size_stride(buf109, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf110 = buf108[1]
        assert_size_stride(buf110, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf111 = buf108[2]
        assert_size_stride(buf111, (), ())
        buf112 = buf108[3]
        assert_size_stride(buf112, (), ())
        del buf108
        buf113 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_46, (4096, 4096), (1, 4096), 0), out=buf113)
        buf114 = reinterpret_tensor(buf113, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf113  # reuse
        buf115 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf116 = reinterpret_tensor(buf115, (1, s0, 1), (s0, 1, 1), 0); del buf115  # reuse
        buf117 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, pow_10, variance_9, add_32, rsqrt_9, hidden_states_47, to_23, hidden_states_48], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf114, buf116, buf98, primals_47, buf117, s0, 4096, stream=stream0)
        buf118 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_48, (4096, 14336), (1, 4096), 0), out=buf118)
        buf119 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_49, (4096, 14336), (1, 4096), 0), out=buf119)
        buf120 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_4, mul_46], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf118, buf119, buf120, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf121 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_50, (14336, 4096), (1, 14336), 0), out=buf121)
        buf122 = reinterpret_tensor(buf121, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf121  # reuse
        buf123 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf124 = reinterpret_tensor(buf123, (1, s0, 1), (s0, 1, 1), 0); del buf123  # reuse
        buf125 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_49, hidden_states_50, pow_11, variance_10, add_34, rsqrt_10, hidden_states_51, to_25, hidden_states_52], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf122, buf124, buf114, primals_51, buf125, s0, 4096, stream=stream0)
        buf126 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_52, (4096, 4096), (1, 4096), 0), out=buf126)
        buf127 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_53, (4096, 1024), (1, 4096), 0), out=buf127)
        buf128 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_54, (4096, 1024), (1, 4096), 0), out=buf128)
        buf129 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf127, buf2, buf129, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf130 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf128, buf130, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf131 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_49, cat_11, mul_50, q_embed_5, query_5], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf126, buf2, buf131, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_20], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf131, reinterpret_tensor(buf129, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf130, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf133 = buf132[0]
        assert_size_stride(buf133, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf134 = buf132[1]
        assert_size_stride(buf134, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf135 = buf132[2]
        assert_size_stride(buf135, (), ())
        buf136 = buf132[3]
        assert_size_stride(buf136, (), ())
        del buf132
        buf137 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_55, (4096, 4096), (1, 4096), 0), out=buf137)
        buf138 = reinterpret_tensor(buf137, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf137  # reuse
        buf139 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf140 = reinterpret_tensor(buf139, (1, s0, 1), (s0, 1, 1), 0); del buf139  # reuse
        buf141 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, pow_12, variance_11, add_38, rsqrt_11, hidden_states_57, to_27, hidden_states_58], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf138, buf140, buf122, primals_56, buf141, s0, 4096, stream=stream0)
        buf142 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_57, (4096, 14336), (1, 4096), 0), out=buf142)
        buf143 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_58, (4096, 14336), (1, 4096), 0), out=buf143)
        buf144 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_5, mul_55], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf142, buf143, buf144, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf145 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_59, (14336, 4096), (1, 14336), 0), out=buf145)
        buf146 = reinterpret_tensor(buf145, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf145  # reuse
        buf147 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf148 = reinterpret_tensor(buf147, (1, s0, 1), (s0, 1, 1), 0); del buf147  # reuse
        buf149 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, pow_13, variance_12, add_40, rsqrt_12, hidden_states_61, to_29, hidden_states_62], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf146, buf148, buf138, primals_60, buf149, s0, 4096, stream=stream0)
        buf150 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_61, (4096, 4096), (1, 4096), 0), out=buf150)
        buf151 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_62, (4096, 1024), (1, 4096), 0), out=buf151)
        buf152 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_63, (4096, 1024), (1, 4096), 0), out=buf152)
        buf153 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf151, buf2, buf153, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf154 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf152, buf154, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf155 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_58, cat_13, mul_59, q_embed_6, query_6], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf150, buf2, buf155, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf155, reinterpret_tensor(buf153, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf154, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf157 = buf156[0]
        assert_size_stride(buf157, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf158 = buf156[1]
        assert_size_stride(buf158, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf159 = buf156[2]
        assert_size_stride(buf159, (), ())
        buf160 = buf156[3]
        assert_size_stride(buf160, (), ())
        del buf156
        buf161 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_64, (4096, 4096), (1, 4096), 0), out=buf161)
        buf162 = reinterpret_tensor(buf161, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf161  # reuse
        buf163 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf164 = reinterpret_tensor(buf163, (1, s0, 1), (s0, 1, 1), 0); del buf163  # reuse
        buf165 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_66, pow_14, variance_13, add_44, rsqrt_13, hidden_states_67, to_31, hidden_states_68], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf162, buf164, buf146, primals_65, buf165, s0, 4096, stream=stream0)
        buf166 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_66, (4096, 14336), (1, 4096), 0), out=buf166)
        buf167 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_67, (4096, 14336), (1, 4096), 0), out=buf167)
        buf168 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_6, mul_64], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf166, buf167, buf168, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf169 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_68, (14336, 4096), (1, 14336), 0), out=buf169)
        buf170 = reinterpret_tensor(buf169, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf169  # reuse
        buf171 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf172 = reinterpret_tensor(buf171, (1, s0, 1), (s0, 1, 1), 0); del buf171  # reuse
        buf173 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_69, hidden_states_70, pow_15, variance_14, add_46, rsqrt_14, hidden_states_71, to_33, hidden_states_72], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf170, buf172, buf162, primals_69, buf173, s0, 4096, stream=stream0)
        buf174 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_70, (4096, 4096), (1, 4096), 0), out=buf174)
        buf175 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_71, (4096, 1024), (1, 4096), 0), out=buf175)
        buf176 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_72, (4096, 1024), (1, 4096), 0), out=buf176)
        buf177 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf175, buf2, buf177, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf178 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf176, buf178, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf179 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_67, cat_15, mul_68, q_embed_7, query_7], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf174, buf2, buf179, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf180 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf179, reinterpret_tensor(buf177, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf178, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf181 = buf180[0]
        assert_size_stride(buf181, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf182 = buf180[1]
        assert_size_stride(buf182, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf183 = buf180[2]
        assert_size_stride(buf183, (), ())
        buf184 = buf180[3]
        assert_size_stride(buf184, (), ())
        del buf180
        buf185 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_73, (4096, 4096), (1, 4096), 0), out=buf185)
        buf186 = reinterpret_tensor(buf185, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf185  # reuse
        buf187 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf188 = reinterpret_tensor(buf187, (1, s0, 1), (s0, 1, 1), 0); del buf187  # reuse
        buf189 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, pow_16, variance_15, add_50, rsqrt_15, hidden_states_77, to_35, hidden_states_78], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf186, buf188, buf170, primals_74, buf189, s0, 4096, stream=stream0)
        buf190 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_75, (4096, 14336), (1, 4096), 0), out=buf190)
        buf191 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_76, (4096, 14336), (1, 4096), 0), out=buf191)
        buf192 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_7, mul_73], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf190, buf191, buf192, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf193 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_77, (14336, 4096), (1, 14336), 0), out=buf193)
        buf194 = reinterpret_tensor(buf193, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf193  # reuse
        buf195 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf196 = reinterpret_tensor(buf195, (1, s0, 1), (s0, 1, 1), 0); del buf195  # reuse
        buf197 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_79, hidden_states_80, pow_17, variance_16, add_52, rsqrt_16, hidden_states_81, to_37, hidden_states_82], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf194, buf196, buf186, primals_78, buf197, s0, 4096, stream=stream0)
        buf198 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_79, (4096, 4096), (1, 4096), 0), out=buf198)
        buf199 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_80, (4096, 1024), (1, 4096), 0), out=buf199)
        buf200 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_81, (4096, 1024), (1, 4096), 0), out=buf200)
        buf201 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf199, buf2, buf201, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf202 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf200, buf202, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf203 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_76, cat_17, mul_77, q_embed_8, query_8], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf198, buf2, buf203, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_32], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf204 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf203, reinterpret_tensor(buf201, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf202, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf205 = buf204[0]
        assert_size_stride(buf205, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf206 = buf204[1]
        assert_size_stride(buf206, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf207 = buf204[2]
        assert_size_stride(buf207, (), ())
        buf208 = buf204[3]
        assert_size_stride(buf208, (), ())
        del buf204
        buf209 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_82, (4096, 4096), (1, 4096), 0), out=buf209)
        buf210 = reinterpret_tensor(buf209, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf209  # reuse
        buf211 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf212 = reinterpret_tensor(buf211, (1, s0, 1), (s0, 1, 1), 0); del buf211  # reuse
        buf213 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, pow_18, variance_17, add_56, rsqrt_17, hidden_states_87, to_39, hidden_states_88], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf210, buf212, buf194, primals_83, buf213, s0, 4096, stream=stream0)
        buf214 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_84, (4096, 14336), (1, 4096), 0), out=buf214)
        buf215 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_85, (4096, 14336), (1, 4096), 0), out=buf215)
        buf216 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_8, mul_82], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf214, buf215, buf216, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf217 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_86, (14336, 4096), (1, 14336), 0), out=buf217)
        buf218 = reinterpret_tensor(buf217, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf217  # reuse
        buf219 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf220 = reinterpret_tensor(buf219, (1, s0, 1), (s0, 1, 1), 0); del buf219  # reuse
        buf221 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, pow_19, variance_18, add_58, rsqrt_18, hidden_states_91, to_41, hidden_states_92], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf218, buf220, buf210, primals_87, buf221, s0, 4096, stream=stream0)
        buf222 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_88, (4096, 4096), (1, 4096), 0), out=buf222)
        buf223 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_89, (4096, 1024), (1, 4096), 0), out=buf223)
        buf224 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_90, (4096, 1024), (1, 4096), 0), out=buf224)
        buf225 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf223, buf2, buf225, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf226 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf224, buf226, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf227 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_85, cat_19, mul_86, q_embed_9, query_9], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf222, buf2, buf227, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_36], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf228 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf227, reinterpret_tensor(buf225, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf226, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf229 = buf228[0]
        assert_size_stride(buf229, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf230 = buf228[1]
        assert_size_stride(buf230, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf231 = buf228[2]
        assert_size_stride(buf231, (), ())
        buf232 = buf228[3]
        assert_size_stride(buf232, (), ())
        del buf228
        buf233 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [attn_output_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_91, (4096, 4096), (1, 4096), 0), out=buf233)
        buf234 = reinterpret_tensor(buf233, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf233  # reuse
        buf235 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf236 = reinterpret_tensor(buf235, (1, s0, 1), (s0, 1, 1), 0); del buf235  # reuse
        buf237 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_95, hidden_states_96, pow_20, variance_19, add_62, rsqrt_19, hidden_states_97, to_43, hidden_states_98], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf234, buf236, buf218, primals_92, buf237, s0, 4096, stream=stream0)
        buf238 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_93, (4096, 14336), (1, 4096), 0), out=buf238)
        buf239 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_94, (4096, 14336), (1, 4096), 0), out=buf239)
        buf240 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_9, mul_91], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf238, buf239, buf240, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf241 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_95, (14336, 4096), (1, 14336), 0), out=buf241)
        buf242 = reinterpret_tensor(buf241, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf241  # reuse
        buf243 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf244 = reinterpret_tensor(buf243, (1, s0, 1), (s0, 1, 1), 0); del buf243  # reuse
        buf245 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_99, hidden_states_100, pow_21, variance_20, add_64, rsqrt_20, hidden_states_101, to_45, hidden_states_102], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf242, buf244, buf234, primals_96, buf245, s0, 4096, stream=stream0)
        buf246 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_97, (4096, 4096), (1, 4096), 0), out=buf246)
        buf247 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_98, (4096, 1024), (1, 4096), 0), out=buf247)
        buf248 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_99, (4096, 1024), (1, 4096), 0), out=buf248)
        buf249 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf247, buf2, buf249, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf250 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf248, buf250, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf251 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_94, cat_21, mul_95, q_embed_10, query_10], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf246, buf2, buf251, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_40], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf252 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf251, reinterpret_tensor(buf249, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf250, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf253 = buf252[0]
        assert_size_stride(buf253, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf254 = buf252[1]
        assert_size_stride(buf254, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf255 = buf252[2]
        assert_size_stride(buf255, (), ())
        buf256 = buf252[3]
        assert_size_stride(buf256, (), ())
        del buf252
        buf257 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_100, (4096, 4096), (1, 4096), 0), out=buf257)
        buf258 = reinterpret_tensor(buf257, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf257  # reuse
        buf259 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf260 = reinterpret_tensor(buf259, (1, s0, 1), (s0, 1, 1), 0); del buf259  # reuse
        buf261 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_105, hidden_states_106, pow_22, variance_21, add_68, rsqrt_21, hidden_states_107, to_47, hidden_states_108], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf258, buf260, buf242, primals_101, buf261, s0, 4096, stream=stream0)
        buf262 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_102, (4096, 14336), (1, 4096), 0), out=buf262)
        buf263 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_103, (4096, 14336), (1, 4096), 0), out=buf263)
        buf264 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_10, mul_100], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf262, buf263, buf264, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf265 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_104, (14336, 4096), (1, 14336), 0), out=buf265)
        buf266 = reinterpret_tensor(buf265, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf265  # reuse
        buf267 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf268 = reinterpret_tensor(buf267, (1, s0, 1), (s0, 1, 1), 0); del buf267  # reuse
        buf269 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110, pow_23, variance_22, add_70, rsqrt_22, hidden_states_111, to_49, hidden_states_112], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf266, buf268, buf258, primals_105, buf269, s0, 4096, stream=stream0)
        buf270 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_106, (4096, 4096), (1, 4096), 0), out=buf270)
        buf271 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_107, (4096, 1024), (1, 4096), 0), out=buf271)
        buf272 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_108, (4096, 1024), (1, 4096), 0), out=buf272)
        buf273 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf271, buf2, buf273, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf274 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf272, buf274, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf275 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_103, cat_23, mul_104, q_embed_11, query_11], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf270, buf2, buf275, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_44], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf276 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf275, reinterpret_tensor(buf273, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf274, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf277 = buf276[0]
        assert_size_stride(buf277, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf278 = buf276[1]
        assert_size_stride(buf278, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf279 = buf276[2]
        assert_size_stride(buf279, (), ())
        buf280 = buf276[3]
        assert_size_stride(buf280, (), ())
        del buf276
        buf281 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [attn_output_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_109, (4096, 4096), (1, 4096), 0), out=buf281)
        buf282 = reinterpret_tensor(buf281, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf281  # reuse
        buf283 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf284 = reinterpret_tensor(buf283, (1, s0, 1), (s0, 1, 1), 0); del buf283  # reuse
        buf285 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116, pow_24, variance_23, add_74, rsqrt_23, hidden_states_117, to_51, hidden_states_118], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf282, buf284, buf266, primals_110, buf285, s0, 4096, stream=stream0)
        buf286 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_111, (4096, 14336), (1, 4096), 0), out=buf286)
        buf287 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_112, (4096, 14336), (1, 4096), 0), out=buf287)
        buf288 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_11, mul_109], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf286, buf287, buf288, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf289 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_113, (14336, 4096), (1, 14336), 0), out=buf289)
        buf290 = reinterpret_tensor(buf289, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf289  # reuse
        buf291 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf292 = reinterpret_tensor(buf291, (1, s0, 1), (s0, 1, 1), 0); del buf291  # reuse
        buf293 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_119, hidden_states_120, pow_25, variance_24, add_76, rsqrt_24, hidden_states_121, to_53, hidden_states_122], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf290, buf292, buf282, primals_114, buf293, s0, 4096, stream=stream0)
        buf294 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_115, (4096, 4096), (1, 4096), 0), out=buf294)
        buf295 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_116, (4096, 1024), (1, 4096), 0), out=buf295)
        buf296 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_117, (4096, 1024), (1, 4096), 0), out=buf296)
        buf297 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf295, buf2, buf297, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf298 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf296, buf298, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf299 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_112, cat_25, mul_113, q_embed_12, query_12], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf294, buf2, buf299, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf300 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf299, reinterpret_tensor(buf297, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf298, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf301 = buf300[0]
        assert_size_stride(buf301, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf302 = buf300[1]
        assert_size_stride(buf302, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf303 = buf300[2]
        assert_size_stride(buf303, (), ())
        buf304 = buf300[3]
        assert_size_stride(buf304, (), ())
        del buf300
        buf305 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [attn_output_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_118, (4096, 4096), (1, 4096), 0), out=buf305)
        buf306 = reinterpret_tensor(buf305, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf305  # reuse
        buf307 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf308 = reinterpret_tensor(buf307, (1, s0, 1), (s0, 1, 1), 0); del buf307  # reuse
        buf309 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_125, hidden_states_126, pow_26, variance_25, add_80, rsqrt_25, hidden_states_127, to_55, hidden_states_128], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf306, buf308, buf290, primals_119, buf309, s0, 4096, stream=stream0)
        buf310 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_120, (4096, 14336), (1, 4096), 0), out=buf310)
        buf311 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_121, (4096, 14336), (1, 4096), 0), out=buf311)
        buf312 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_12, mul_118], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf310, buf311, buf312, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf313 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_122, (14336, 4096), (1, 14336), 0), out=buf313)
        buf314 = reinterpret_tensor(buf313, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf313  # reuse
        buf315 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf316 = reinterpret_tensor(buf315, (1, s0, 1), (s0, 1, 1), 0); del buf315  # reuse
        buf317 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130, pow_27, variance_26, add_82, rsqrt_26, hidden_states_131, to_57, hidden_states_132], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf314, buf316, buf306, primals_123, buf317, s0, 4096, stream=stream0)
        buf318 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_124, (4096, 4096), (1, 4096), 0), out=buf318)
        buf319 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_125, (4096, 1024), (1, 4096), 0), out=buf319)
        buf320 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_126, (4096, 1024), (1, 4096), 0), out=buf320)
        buf321 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf319, buf2, buf321, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf322 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf320, buf322, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf323 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_121, cat_27, mul_122, q_embed_13, query_13], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf318, buf2, buf323, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_52], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf324 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf323, reinterpret_tensor(buf321, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf322, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf325 = buf324[0]
        assert_size_stride(buf325, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf326 = buf324[1]
        assert_size_stride(buf326, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf327 = buf324[2]
        assert_size_stride(buf327, (), ())
        buf328 = buf324[3]
        assert_size_stride(buf328, (), ())
        del buf324
        buf329 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [attn_output_55], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_127, (4096, 4096), (1, 4096), 0), out=buf329)
        buf330 = reinterpret_tensor(buf329, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf329  # reuse
        buf331 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf332 = reinterpret_tensor(buf331, (1, s0, 1), (s0, 1, 1), 0); del buf331  # reuse
        buf333 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136, pow_28, variance_27, add_86, rsqrt_27, hidden_states_137, to_59, hidden_states_138], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf330, buf332, buf314, primals_128, buf333, s0, 4096, stream=stream0)
        buf334 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_129, (4096, 14336), (1, 4096), 0), out=buf334)
        buf335 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_130, (4096, 14336), (1, 4096), 0), out=buf335)
        buf336 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_13, mul_127], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf334, buf335, buf336, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf337 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_131, (14336, 4096), (1, 14336), 0), out=buf337)
        buf338 = reinterpret_tensor(buf337, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf337  # reuse
        buf339 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf340 = reinterpret_tensor(buf339, (1, s0, 1), (s0, 1, 1), 0); del buf339  # reuse
        buf341 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_139, hidden_states_140, pow_29, variance_28, add_88, rsqrt_28, hidden_states_141, to_61, hidden_states_142], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf338, buf340, buf330, primals_132, buf341, s0, 4096, stream=stream0)
        buf342 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_133, (4096, 4096), (1, 4096), 0), out=buf342)
        buf343 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_134, (4096, 1024), (1, 4096), 0), out=buf343)
        buf344 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_135, (4096, 1024), (1, 4096), 0), out=buf344)
        buf345 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf343, buf2, buf345, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf346 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf344, buf346, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf347 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_130, cat_29, mul_131, q_embed_14, query_14], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf342, buf2, buf347, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_56], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf347, reinterpret_tensor(buf345, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf346, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf349 = buf348[0]
        assert_size_stride(buf349, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf350 = buf348[1]
        assert_size_stride(buf350, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf351 = buf348[2]
        assert_size_stride(buf351, (), ())
        buf352 = buf348[3]
        assert_size_stride(buf352, (), ())
        del buf348
        buf353 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [attn_output_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_136, (4096, 4096), (1, 4096), 0), out=buf353)
        buf354 = reinterpret_tensor(buf353, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf353  # reuse
        buf355 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf356 = reinterpret_tensor(buf355, (1, s0, 1), (s0, 1, 1), 0); del buf355  # reuse
        buf357 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, pow_30, variance_29, add_92, rsqrt_29, hidden_states_147, to_63, hidden_states_148], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf354, buf356, buf338, primals_137, buf357, s0, 4096, stream=stream0)
        buf358 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_138, (4096, 14336), (1, 4096), 0), out=buf358)
        buf359 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_139, (4096, 14336), (1, 4096), 0), out=buf359)
        buf360 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_14, mul_136], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf358, buf359, buf360, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf361 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_140, (14336, 4096), (1, 14336), 0), out=buf361)
        buf362 = reinterpret_tensor(buf361, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf361  # reuse
        buf363 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf364 = reinterpret_tensor(buf363, (1, s0, 1), (s0, 1, 1), 0); del buf363  # reuse
        buf365 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_149, hidden_states_150, pow_31, variance_30, add_94, rsqrt_30, hidden_states_151, to_65, hidden_states_152], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf362, buf364, buf354, primals_141, buf365, s0, 4096, stream=stream0)
        buf366 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_142, (4096, 4096), (1, 4096), 0), out=buf366)
        buf367 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_143, (4096, 1024), (1, 4096), 0), out=buf367)
        buf368 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_144, (4096, 1024), (1, 4096), 0), out=buf368)
        buf369 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf367, buf2, buf369, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf370 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf368, buf370, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf371 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_139, cat_31, mul_140, q_embed_15, query_15], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf366, buf2, buf371, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_60], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf372 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf371, reinterpret_tensor(buf369, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf370, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf373 = buf372[0]
        assert_size_stride(buf373, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf374 = buf372[1]
        assert_size_stride(buf374, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf375 = buf372[2]
        assert_size_stride(buf375, (), ())
        buf376 = buf372[3]
        assert_size_stride(buf376, (), ())
        del buf372
        buf377 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_145, (4096, 4096), (1, 4096), 0), out=buf377)
        buf378 = reinterpret_tensor(buf377, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf377  # reuse
        buf379 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf380 = reinterpret_tensor(buf379, (1, s0, 1), (s0, 1, 1), 0); del buf379  # reuse
        buf381 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_155, hidden_states_156, pow_32, variance_31, add_98, rsqrt_31, hidden_states_157, to_67, hidden_states_158], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf378, buf380, buf362, primals_146, buf381, s0, 4096, stream=stream0)
        buf382 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_147, (4096, 14336), (1, 4096), 0), out=buf382)
        buf383 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_148, (4096, 14336), (1, 4096), 0), out=buf383)
        buf384 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_15, mul_145], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf382, buf383, buf384, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf385 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_149, (14336, 4096), (1, 14336), 0), out=buf385)
        buf386 = reinterpret_tensor(buf385, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf385  # reuse
        buf387 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf388 = reinterpret_tensor(buf387, (1, s0, 1), (s0, 1, 1), 0); del buf387  # reuse
        buf389 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, pow_33, variance_32, add_100, rsqrt_32, hidden_states_161, to_69, hidden_states_162], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf386, buf388, buf378, primals_150, buf389, s0, 4096, stream=stream0)
        buf390 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_151, (4096, 4096), (1, 4096), 0), out=buf390)
        buf391 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_152, (4096, 1024), (1, 4096), 0), out=buf391)
        buf392 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_153, (4096, 1024), (1, 4096), 0), out=buf392)
        buf393 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf391, buf2, buf393, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf394 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf392, buf394, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf395 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_148, cat_33, mul_149, q_embed_16, query_16], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf390, buf2, buf395, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_64], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf396 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf395, reinterpret_tensor(buf393, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf394, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf397 = buf396[0]
        assert_size_stride(buf397, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf398 = buf396[1]
        assert_size_stride(buf398, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf399 = buf396[2]
        assert_size_stride(buf399, (), ())
        buf400 = buf396[3]
        assert_size_stride(buf400, (), ())
        del buf396
        buf401 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [attn_output_67], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_154, (4096, 4096), (1, 4096), 0), out=buf401)
        buf402 = reinterpret_tensor(buf401, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf401  # reuse
        buf403 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf404 = reinterpret_tensor(buf403, (1, s0, 1), (s0, 1, 1), 0); del buf403  # reuse
        buf405 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_165, hidden_states_166, pow_34, variance_33, add_104, rsqrt_33, hidden_states_167, to_71, hidden_states_168], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf402, buf404, buf386, primals_155, buf405, s0, 4096, stream=stream0)
        buf406 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_156, (4096, 14336), (1, 4096), 0), out=buf406)
        buf407 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_157, (4096, 14336), (1, 4096), 0), out=buf407)
        buf408 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_16, mul_154], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf406, buf407, buf408, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf409 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_158, (14336, 4096), (1, 14336), 0), out=buf409)
        buf410 = reinterpret_tensor(buf409, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf409  # reuse
        buf411 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf412 = reinterpret_tensor(buf411, (1, s0, 1), (s0, 1, 1), 0); del buf411  # reuse
        buf413 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_169, hidden_states_170, pow_35, variance_34, add_106, rsqrt_34, hidden_states_171, to_73, hidden_states_172], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf410, buf412, buf402, primals_159, buf413, s0, 4096, stream=stream0)
        buf414 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_160, (4096, 4096), (1, 4096), 0), out=buf414)
        buf415 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_161, (4096, 1024), (1, 4096), 0), out=buf415)
        buf416 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_162, (4096, 1024), (1, 4096), 0), out=buf416)
        buf417 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf415, buf2, buf417, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf418 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf416, buf418, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf419 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_157, cat_35, mul_158, q_embed_17, query_17], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf414, buf2, buf419, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_68], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf420 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf419, reinterpret_tensor(buf417, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf418, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf421 = buf420[0]
        assert_size_stride(buf421, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf422 = buf420[1]
        assert_size_stride(buf422, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf423 = buf420[2]
        assert_size_stride(buf423, (), ())
        buf424 = buf420[3]
        assert_size_stride(buf424, (), ())
        del buf420
        buf425 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [attn_output_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_163, (4096, 4096), (1, 4096), 0), out=buf425)
        buf426 = reinterpret_tensor(buf425, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf425  # reuse
        buf427 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf428 = reinterpret_tensor(buf427, (1, s0, 1), (s0, 1, 1), 0); del buf427  # reuse
        buf429 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_175, hidden_states_176, pow_36, variance_35, add_110, rsqrt_35, hidden_states_177, to_75, hidden_states_178], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf426, buf428, buf410, primals_164, buf429, s0, 4096, stream=stream0)
        buf430 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_165, (4096, 14336), (1, 4096), 0), out=buf430)
        buf431 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_166, (4096, 14336), (1, 4096), 0), out=buf431)
        buf432 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_17, mul_163], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf430, buf431, buf432, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf433 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_167, (14336, 4096), (1, 14336), 0), out=buf433)
        buf434 = reinterpret_tensor(buf433, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf433  # reuse
        buf435 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf436 = reinterpret_tensor(buf435, (1, s0, 1), (s0, 1, 1), 0); del buf435  # reuse
        buf437 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_179, hidden_states_180, pow_37, variance_36, add_112, rsqrt_36, hidden_states_181, to_77, hidden_states_182], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf434, buf436, buf426, primals_168, buf437, s0, 4096, stream=stream0)
        buf438 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_169, (4096, 4096), (1, 4096), 0), out=buf438)
        buf439 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_170, (4096, 1024), (1, 4096), 0), out=buf439)
        buf440 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_171, (4096, 1024), (1, 4096), 0), out=buf440)
        buf441 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf439, buf2, buf441, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf442 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf440, buf442, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf443 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_166, cat_37, mul_167, q_embed_18, query_18], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf438, buf2, buf443, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_72], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf444 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf443, reinterpret_tensor(buf441, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf442, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf445 = buf444[0]
        assert_size_stride(buf445, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf446 = buf444[1]
        assert_size_stride(buf446, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf447 = buf444[2]
        assert_size_stride(buf447, (), ())
        buf448 = buf444[3]
        assert_size_stride(buf448, (), ())
        del buf444
        buf449 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [attn_output_75], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_172, (4096, 4096), (1, 4096), 0), out=buf449)
        buf450 = reinterpret_tensor(buf449, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf449  # reuse
        buf451 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf452 = reinterpret_tensor(buf451, (1, s0, 1), (s0, 1, 1), 0); del buf451  # reuse
        buf453 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_185, hidden_states_186, pow_38, variance_37, add_116, rsqrt_37, hidden_states_187, to_79, hidden_states_188], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf450, buf452, buf434, primals_173, buf453, s0, 4096, stream=stream0)
        buf454 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_174, (4096, 14336), (1, 4096), 0), out=buf454)
        buf455 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_175, (4096, 14336), (1, 4096), 0), out=buf455)
        buf456 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_18, mul_172], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf454, buf455, buf456, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf457 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_176, (14336, 4096), (1, 14336), 0), out=buf457)
        buf458 = reinterpret_tensor(buf457, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf457  # reuse
        buf459 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf460 = reinterpret_tensor(buf459, (1, s0, 1), (s0, 1, 1), 0); del buf459  # reuse
        buf461 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_189, hidden_states_190, pow_39, variance_38, add_118, rsqrt_38, hidden_states_191, to_81, hidden_states_192], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf458, buf460, buf450, primals_177, buf461, s0, 4096, stream=stream0)
        buf462 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_178, (4096, 4096), (1, 4096), 0), out=buf462)
        buf463 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_179, (4096, 1024), (1, 4096), 0), out=buf463)
        buf464 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_180, (4096, 1024), (1, 4096), 0), out=buf464)
        buf465 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf463, buf2, buf465, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf466 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf464, buf466, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf467 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_175, cat_39, mul_176, q_embed_19, query_19], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf462, buf2, buf467, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_76], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf468 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf467, reinterpret_tensor(buf465, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf466, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf469 = buf468[0]
        assert_size_stride(buf469, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf470 = buf468[1]
        assert_size_stride(buf470, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf471 = buf468[2]
        assert_size_stride(buf471, (), ())
        buf472 = buf468[3]
        assert_size_stride(buf472, (), ())
        del buf468
        buf473 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [attn_output_79], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_181, (4096, 4096), (1, 4096), 0), out=buf473)
        buf474 = reinterpret_tensor(buf473, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf473  # reuse
        buf475 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf476 = reinterpret_tensor(buf475, (1, s0, 1), (s0, 1, 1), 0); del buf475  # reuse
        buf477 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_195, hidden_states_196, pow_40, variance_39, add_122, rsqrt_39, hidden_states_197, to_83, hidden_states_198], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf474, buf476, buf458, primals_182, buf477, s0, 4096, stream=stream0)
        buf478 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_183, (4096, 14336), (1, 4096), 0), out=buf478)
        buf479 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_184, (4096, 14336), (1, 4096), 0), out=buf479)
        buf480 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_19, mul_181], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf478, buf479, buf480, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf481 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_185, (14336, 4096), (1, 14336), 0), out=buf481)
        buf482 = reinterpret_tensor(buf481, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf481  # reuse
        buf483 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf484 = reinterpret_tensor(buf483, (1, s0, 1), (s0, 1, 1), 0); del buf483  # reuse
        buf485 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_200, pow_41, variance_40, add_124, rsqrt_40, hidden_states_201, to_85, hidden_states_202], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf482, buf484, buf474, primals_186, buf485, s0, 4096, stream=stream0)
        buf486 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_187, (4096, 4096), (1, 4096), 0), out=buf486)
        buf487 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_188, (4096, 1024), (1, 4096), 0), out=buf487)
        buf488 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_189, (4096, 1024), (1, 4096), 0), out=buf488)
        buf489 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf487, buf2, buf489, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf490 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf488, buf490, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf491 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_184, cat_41, mul_185, q_embed_20, query_20], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf486, buf2, buf491, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_80], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf492 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf491, reinterpret_tensor(buf489, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf490, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf493 = buf492[0]
        assert_size_stride(buf493, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf494 = buf492[1]
        assert_size_stride(buf494, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf495 = buf492[2]
        assert_size_stride(buf495, (), ())
        buf496 = buf492[3]
        assert_size_stride(buf496, (), ())
        del buf492
        buf497 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [attn_output_83], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_190, (4096, 4096), (1, 4096), 0), out=buf497)
        buf498 = reinterpret_tensor(buf497, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf497  # reuse
        buf499 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf500 = reinterpret_tensor(buf499, (1, s0, 1), (s0, 1, 1), 0); del buf499  # reuse
        buf501 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_205, hidden_states_206, pow_42, variance_41, add_128, rsqrt_41, hidden_states_207, to_87, hidden_states_208], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf498, buf500, buf482, primals_191, buf501, s0, 4096, stream=stream0)
        buf502 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_192, (4096, 14336), (1, 4096), 0), out=buf502)
        buf503 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_193, (4096, 14336), (1, 4096), 0), out=buf503)
        buf504 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_20, mul_190], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf502, buf503, buf504, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf505 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_194, (14336, 4096), (1, 14336), 0), out=buf505)
        buf506 = reinterpret_tensor(buf505, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf505  # reuse
        buf507 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf508 = reinterpret_tensor(buf507, (1, s0, 1), (s0, 1, 1), 0); del buf507  # reuse
        buf509 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_209, hidden_states_210, pow_43, variance_42, add_130, rsqrt_42, hidden_states_211, to_89, hidden_states_212], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf506, buf508, buf498, primals_195, buf509, s0, 4096, stream=stream0)
        buf510 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_196, (4096, 4096), (1, 4096), 0), out=buf510)
        buf511 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_197, (4096, 1024), (1, 4096), 0), out=buf511)
        buf512 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_198, (4096, 1024), (1, 4096), 0), out=buf512)
        buf513 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf511, buf2, buf513, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf514 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf512, buf514, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf515 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_193, cat_43, mul_194, q_embed_21, query_21], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf510, buf2, buf515, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_84], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf516 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf515, reinterpret_tensor(buf513, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf514, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf517 = buf516[0]
        assert_size_stride(buf517, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf518 = buf516[1]
        assert_size_stride(buf518, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf519 = buf516[2]
        assert_size_stride(buf519, (), ())
        buf520 = buf516[3]
        assert_size_stride(buf520, (), ())
        del buf516
        buf521 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [attn_output_87], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf517, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_199, (4096, 4096), (1, 4096), 0), out=buf521)
        buf522 = reinterpret_tensor(buf521, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf521  # reuse
        buf523 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf524 = reinterpret_tensor(buf523, (1, s0, 1), (s0, 1, 1), 0); del buf523  # reuse
        buf525 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_215, hidden_states_216, pow_44, variance_43, add_134, rsqrt_43, hidden_states_217, to_91, hidden_states_218], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf522, buf524, buf506, primals_200, buf525, s0, 4096, stream=stream0)
        buf526 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_151], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_201, (4096, 14336), (1, 4096), 0), out=buf526)
        buf527 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_202, (4096, 14336), (1, 4096), 0), out=buf527)
        buf528 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_21, mul_199], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf526, buf527, buf528, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf529 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf528, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_203, (14336, 4096), (1, 14336), 0), out=buf529)
        buf530 = reinterpret_tensor(buf529, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf529  # reuse
        buf531 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf532 = reinterpret_tensor(buf531, (1, s0, 1), (s0, 1, 1), 0); del buf531  # reuse
        buf533 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_219, hidden_states_220, pow_45, variance_44, add_136, rsqrt_44, hidden_states_221, to_93, hidden_states_222], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf530, buf532, buf522, primals_204, buf533, s0, 4096, stream=stream0)
        buf534 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_205, (4096, 4096), (1, 4096), 0), out=buf534)
        buf535 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_206, (4096, 1024), (1, 4096), 0), out=buf535)
        buf536 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_207, (4096, 1024), (1, 4096), 0), out=buf536)
        buf537 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf535, buf2, buf537, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf538 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf536, buf538, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf539 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_202, cat_45, mul_203, q_embed_22, query_22], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf534, buf2, buf539, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_88], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf540 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf539, reinterpret_tensor(buf537, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf538, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf541 = buf540[0]
        assert_size_stride(buf541, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf542 = buf540[1]
        assert_size_stride(buf542, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf543 = buf540[2]
        assert_size_stride(buf543, (), ())
        buf544 = buf540[3]
        assert_size_stride(buf544, (), ())
        del buf540
        buf545 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [attn_output_91], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_208, (4096, 4096), (1, 4096), 0), out=buf545)
        buf546 = reinterpret_tensor(buf545, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf545  # reuse
        buf547 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf548 = reinterpret_tensor(buf547, (1, s0, 1), (s0, 1, 1), 0); del buf547  # reuse
        buf549 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_225, hidden_states_226, pow_46, variance_45, add_140, rsqrt_45, hidden_states_227, to_95, hidden_states_228], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf546, buf548, buf530, primals_209, buf549, s0, 4096, stream=stream0)
        buf550 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_210, (4096, 14336), (1, 4096), 0), out=buf550)
        buf551 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_211, (4096, 14336), (1, 4096), 0), out=buf551)
        buf552 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_22, mul_208], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf550, buf551, buf552, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf553 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_212, (14336, 4096), (1, 14336), 0), out=buf553)
        buf554 = reinterpret_tensor(buf553, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf553  # reuse
        buf555 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf556 = reinterpret_tensor(buf555, (1, s0, 1), (s0, 1, 1), 0); del buf555  # reuse
        buf557 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_229, hidden_states_230, pow_47, variance_46, add_142, rsqrt_46, hidden_states_231, to_97, hidden_states_232], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf554, buf556, buf546, primals_213, buf557, s0, 4096, stream=stream0)
        buf558 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_214, (4096, 4096), (1, 4096), 0), out=buf558)
        buf559 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_215, (4096, 1024), (1, 4096), 0), out=buf559)
        buf560 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_216, (4096, 1024), (1, 4096), 0), out=buf560)
        buf561 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf559, buf2, buf561, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf562 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf560, buf562, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf563 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_211, cat_47, mul_212, q_embed_23, query_23], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf558, buf2, buf563, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_92], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf564 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf563, reinterpret_tensor(buf561, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf562, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf565 = buf564[0]
        assert_size_stride(buf565, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf566 = buf564[1]
        assert_size_stride(buf566, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf567 = buf564[2]
        assert_size_stride(buf567, (), ())
        buf568 = buf564[3]
        assert_size_stride(buf568, (), ())
        del buf564
        buf569 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [attn_output_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_217, (4096, 4096), (1, 4096), 0), out=buf569)
        buf570 = reinterpret_tensor(buf569, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf569  # reuse
        buf571 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf572 = reinterpret_tensor(buf571, (1, s0, 1), (s0, 1, 1), 0); del buf571  # reuse
        buf573 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_235, hidden_states_236, pow_48, variance_47, add_146, rsqrt_47, hidden_states_237, to_99, hidden_states_238], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf570, buf572, buf554, primals_218, buf573, s0, 4096, stream=stream0)
        buf574 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_165], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_219, (4096, 14336), (1, 4096), 0), out=buf574)
        buf575 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_220, (4096, 14336), (1, 4096), 0), out=buf575)
        buf576 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_23, mul_217], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf574, buf575, buf576, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf577 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_221, (14336, 4096), (1, 14336), 0), out=buf577)
        buf578 = reinterpret_tensor(buf577, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf577  # reuse
        buf579 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf580 = reinterpret_tensor(buf579, (1, s0, 1), (s0, 1, 1), 0); del buf579  # reuse
        buf581 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_239, hidden_states_240, pow_49, variance_48, add_148, rsqrt_48, hidden_states_241, to_101, hidden_states_242], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf578, buf580, buf570, primals_222, buf581, s0, 4096, stream=stream0)
        buf582 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_168], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_223, (4096, 4096), (1, 4096), 0), out=buf582)
        buf583 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_224, (4096, 1024), (1, 4096), 0), out=buf583)
        buf584 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_225, (4096, 1024), (1, 4096), 0), out=buf584)
        buf585 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf583, buf2, buf585, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf586 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf584, buf586, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf587 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_220, cat_49, mul_221, q_embed_24, query_24], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf582, buf2, buf587, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_96], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf588 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf587, reinterpret_tensor(buf585, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf586, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf589 = buf588[0]
        assert_size_stride(buf589, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf590 = buf588[1]
        assert_size_stride(buf590, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf591 = buf588[2]
        assert_size_stride(buf591, (), ())
        buf592 = buf588[3]
        assert_size_stride(buf592, (), ())
        del buf588
        buf593 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [attn_output_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf589, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_226, (4096, 4096), (1, 4096), 0), out=buf593)
        buf594 = reinterpret_tensor(buf593, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf593  # reuse
        buf595 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf596 = reinterpret_tensor(buf595, (1, s0, 1), (s0, 1, 1), 0); del buf595  # reuse
        buf597 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_245, hidden_states_246, pow_50, variance_49, add_152, rsqrt_49, hidden_states_247, to_103, hidden_states_248], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf594, buf596, buf578, primals_227, buf597, s0, 4096, stream=stream0)
        buf598 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_172], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_228, (4096, 14336), (1, 4096), 0), out=buf598)
        buf599 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_229, (4096, 14336), (1, 4096), 0), out=buf599)
        buf600 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_24, mul_226], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf598, buf599, buf600, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf601 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_230, (14336, 4096), (1, 14336), 0), out=buf601)
        buf602 = reinterpret_tensor(buf601, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf601  # reuse
        buf603 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf604 = reinterpret_tensor(buf603, (1, s0, 1), (s0, 1, 1), 0); del buf603  # reuse
        buf605 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_249, hidden_states_250, pow_51, variance_50, add_154, rsqrt_50, hidden_states_251, to_105, hidden_states_252], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf602, buf604, buf594, primals_231, buf605, s0, 4096, stream=stream0)
        buf606 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf605, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_232, (4096, 4096), (1, 4096), 0), out=buf606)
        buf607 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf605, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_233, (4096, 1024), (1, 4096), 0), out=buf607)
        buf608 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [linear_177], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf605, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_234, (4096, 1024), (1, 4096), 0), out=buf608)
        buf609 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf607, buf2, buf609, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf610 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf608, buf610, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf611 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_229, cat_51, mul_230, q_embed_25, query_25], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf606, buf2, buf611, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_100], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf612 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf611, reinterpret_tensor(buf609, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf610, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf613 = buf612[0]
        assert_size_stride(buf613, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf614 = buf612[1]
        assert_size_stride(buf614, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf615 = buf612[2]
        assert_size_stride(buf615, (), ())
        buf616 = buf612[3]
        assert_size_stride(buf616, (), ())
        del buf612
        buf617 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [attn_output_103], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_235, (4096, 4096), (1, 4096), 0), out=buf617)
        buf618 = reinterpret_tensor(buf617, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf617  # reuse
        buf619 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf620 = reinterpret_tensor(buf619, (1, s0, 1), (s0, 1, 1), 0); del buf619  # reuse
        buf621 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_255, hidden_states_256, pow_52, variance_51, add_158, rsqrt_51, hidden_states_257, to_107, hidden_states_258], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf618, buf620, buf602, primals_236, buf621, s0, 4096, stream=stream0)
        buf622 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_179], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_237, (4096, 14336), (1, 4096), 0), out=buf622)
        buf623 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_180], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_238, (4096, 14336), (1, 4096), 0), out=buf623)
        buf624 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_25, mul_235], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf622, buf623, buf624, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf625 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_239, (14336, 4096), (1, 14336), 0), out=buf625)
        buf626 = reinterpret_tensor(buf625, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf625  # reuse
        buf627 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf628 = reinterpret_tensor(buf627, (1, s0, 1), (s0, 1, 1), 0); del buf627  # reuse
        buf629 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_259, hidden_states_260, pow_53, variance_52, add_160, rsqrt_52, hidden_states_261, to_109, hidden_states_262], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf626, buf628, buf618, primals_240, buf629, s0, 4096, stream=stream0)
        buf630 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_241, (4096, 4096), (1, 4096), 0), out=buf630)
        buf631 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [linear_183], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_242, (4096, 1024), (1, 4096), 0), out=buf631)
        buf632 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [linear_184], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_243, (4096, 1024), (1, 4096), 0), out=buf632)
        buf633 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf631, buf2, buf633, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf634 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf632, buf634, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf635 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_238, cat_53, mul_239, q_embed_26, query_26], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf630, buf2, buf635, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_104], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf636 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf635, reinterpret_tensor(buf633, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf634, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf637 = buf636[0]
        assert_size_stride(buf637, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf638 = buf636[1]
        assert_size_stride(buf638, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf639 = buf636[2]
        assert_size_stride(buf639, (), ())
        buf640 = buf636[3]
        assert_size_stride(buf640, (), ())
        del buf636
        buf641 = buf630; del buf630  # reuse
        # Topologically Sorted Source Nodes: [attn_output_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_244, (4096, 4096), (1, 4096), 0), out=buf641)
        buf642 = reinterpret_tensor(buf641, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf641  # reuse
        buf643 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf644 = reinterpret_tensor(buf643, (1, s0, 1), (s0, 1, 1), 0); del buf643  # reuse
        buf645 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_265, hidden_states_266, pow_54, variance_53, add_164, rsqrt_53, hidden_states_267, to_111, hidden_states_268], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf642, buf644, buf626, primals_245, buf645, s0, 4096, stream=stream0)
        buf646 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_186], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf645, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_246, (4096, 14336), (1, 4096), 0), out=buf646)
        buf647 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf645, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_247, (4096, 14336), (1, 4096), 0), out=buf647)
        buf648 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_26, mul_244], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf646, buf647, buf648, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf649 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_248, (14336, 4096), (1, 14336), 0), out=buf649)
        buf650 = reinterpret_tensor(buf649, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf649  # reuse
        buf651 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf652 = reinterpret_tensor(buf651, (1, s0, 1), (s0, 1, 1), 0); del buf651  # reuse
        buf653 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_269, hidden_states_270, pow_55, variance_54, add_166, rsqrt_54, hidden_states_271, to_113, hidden_states_272], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf650, buf652, buf642, primals_249, buf653, s0, 4096, stream=stream0)
        buf654 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_189], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_250, (4096, 4096), (1, 4096), 0), out=buf654)
        buf655 = buf632; del buf632  # reuse
        # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_251, (4096, 1024), (1, 4096), 0), out=buf655)
        buf656 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_252, (4096, 1024), (1, 4096), 0), out=buf656)
        buf657 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf655, buf2, buf657, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf658 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf656, buf658, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf659 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_247, cat_55, mul_248, q_embed_27, query_27], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf654, buf2, buf659, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_108], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf660 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf659, reinterpret_tensor(buf657, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf658, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf661 = buf660[0]
        assert_size_stride(buf661, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf662 = buf660[1]
        assert_size_stride(buf662, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf663 = buf660[2]
        assert_size_stride(buf663, (), ())
        buf664 = buf660[3]
        assert_size_stride(buf664, (), ())
        del buf660
        buf665 = buf654; del buf654  # reuse
        # Topologically Sorted Source Nodes: [attn_output_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf661, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_253, (4096, 4096), (1, 4096), 0), out=buf665)
        buf666 = reinterpret_tensor(buf665, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf665  # reuse
        buf667 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf668 = reinterpret_tensor(buf667, (1, s0, 1), (s0, 1, 1), 0); del buf667  # reuse
        buf669 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_275, hidden_states_276, pow_56, variance_55, add_170, rsqrt_55, hidden_states_277, to_115, hidden_states_278], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf666, buf668, buf650, primals_254, buf669, s0, 4096, stream=stream0)
        buf670 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_193], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_255, (4096, 14336), (1, 4096), 0), out=buf670)
        buf671 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_256, (4096, 14336), (1, 4096), 0), out=buf671)
        buf672 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_27, mul_253], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf670, buf671, buf672, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf673 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf672, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_257, (14336, 4096), (1, 14336), 0), out=buf673)
        buf674 = reinterpret_tensor(buf673, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf673  # reuse
        buf675 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf676 = reinterpret_tensor(buf675, (1, s0, 1), (s0, 1, 1), 0); del buf675  # reuse
        buf677 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_279, hidden_states_280, pow_57, variance_56, add_172, rsqrt_56, hidden_states_281, to_117, hidden_states_282], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf674, buf676, buf666, primals_258, buf677, s0, 4096, stream=stream0)
        buf678 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_196], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_259, (4096, 4096), (1, 4096), 0), out=buf678)
        buf679 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [linear_197], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_260, (4096, 1024), (1, 4096), 0), out=buf679)
        buf680 = buf655; del buf655  # reuse
        # Topologically Sorted Source Nodes: [linear_198], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_261, (4096, 1024), (1, 4096), 0), out=buf680)
        buf681 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf679, buf2, buf681, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf682 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf680, buf682, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf683 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_256, cat_57, mul_257, q_embed_28, query_28], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf678, buf2, buf683, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_112], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf684 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf683, reinterpret_tensor(buf681, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf682, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf685 = buf684[0]
        assert_size_stride(buf685, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf686 = buf684[1]
        assert_size_stride(buf686, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf687 = buf684[2]
        assert_size_stride(buf687, (), ())
        buf688 = buf684[3]
        assert_size_stride(buf688, (), ())
        del buf684
        buf689 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [attn_output_115], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_262, (4096, 4096), (1, 4096), 0), out=buf689)
        buf690 = reinterpret_tensor(buf689, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf689  # reuse
        buf691 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf692 = reinterpret_tensor(buf691, (1, s0, 1), (s0, 1, 1), 0); del buf691  # reuse
        buf693 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_285, hidden_states_286, pow_58, variance_57, add_176, rsqrt_57, hidden_states_287, to_119, hidden_states_288], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf690, buf692, buf674, primals_263, buf693, s0, 4096, stream=stream0)
        buf694 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_200], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_264, (4096, 14336), (1, 4096), 0), out=buf694)
        buf695 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_201], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_265, (4096, 14336), (1, 4096), 0), out=buf695)
        buf696 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_28, mul_262], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf694, buf695, buf696, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf697 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_266, (14336, 4096), (1, 14336), 0), out=buf697)
        buf698 = reinterpret_tensor(buf697, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf697  # reuse
        buf699 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf700 = reinterpret_tensor(buf699, (1, s0, 1), (s0, 1, 1), 0); del buf699  # reuse
        buf701 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_289, hidden_states_290, pow_59, variance_58, add_178, rsqrt_58, hidden_states_291, to_121, hidden_states_292], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf698, buf700, buf690, primals_267, buf701, s0, 4096, stream=stream0)
        buf702 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_203], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_268, (4096, 4096), (1, 4096), 0), out=buf702)
        buf703 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [linear_204], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_269, (4096, 1024), (1, 4096), 0), out=buf703)
        buf704 = buf679; del buf679  # reuse
        # Topologically Sorted Source Nodes: [linear_205], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_270, (4096, 1024), (1, 4096), 0), out=buf704)
        buf705 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf703, buf2, buf705, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf706 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf704, buf706, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf707 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_265, cat_59, mul_266, q_embed_29, query_29], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf702, buf2, buf707, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_116], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf708 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf707, reinterpret_tensor(buf705, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf706, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf709 = buf708[0]
        assert_size_stride(buf709, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf710 = buf708[1]
        assert_size_stride(buf710, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf711 = buf708[2]
        assert_size_stride(buf711, (), ())
        buf712 = buf708[3]
        assert_size_stride(buf712, (), ())
        del buf708
        buf713 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [attn_output_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf709, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_271, (4096, 4096), (1, 4096), 0), out=buf713)
        buf714 = reinterpret_tensor(buf713, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf713  # reuse
        buf715 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf716 = reinterpret_tensor(buf715, (1, s0, 1), (s0, 1, 1), 0); del buf715  # reuse
        buf717 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_295, hidden_states_296, pow_60, variance_59, add_182, rsqrt_59, hidden_states_297, to_123, hidden_states_298], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf714, buf716, buf698, primals_272, buf717, s0, 4096, stream=stream0)
        buf718 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_207], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf717, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_273, (4096, 14336), (1, 4096), 0), out=buf718)
        buf719 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_208], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf717, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_274, (4096, 14336), (1, 4096), 0), out=buf719)
        buf720 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_29, mul_271], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf718, buf719, buf720, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf721 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_275, (14336, 4096), (1, 14336), 0), out=buf721)
        buf722 = reinterpret_tensor(buf721, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf721  # reuse
        buf723 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf724 = reinterpret_tensor(buf723, (1, s0, 1), (s0, 1, 1), 0); del buf723  # reuse
        buf725 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_299, hidden_states_300, pow_61, variance_60, add_184, rsqrt_60, hidden_states_301, to_125, hidden_states_302], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf722, buf724, buf714, primals_276, buf725, s0, 4096, stream=stream0)
        buf726 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_210], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_277, (4096, 4096), (1, 4096), 0), out=buf726)
        buf727 = buf704; del buf704  # reuse
        # Topologically Sorted Source Nodes: [linear_211], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_278, (4096, 1024), (1, 4096), 0), out=buf727)
        buf728 = buf703; del buf703  # reuse
        # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_279, (4096, 1024), (1, 4096), 0), out=buf728)
        buf729 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf727, buf2, buf729, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        buf730 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf728, buf730, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        buf731 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_274, cat_61, mul_275, q_embed_30, query_30], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf726, buf2, buf731, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_120], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf732 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf731, reinterpret_tensor(buf729, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf730, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf733 = buf732[0]
        assert_size_stride(buf733, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf734 = buf732[1]
        assert_size_stride(buf734, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf735 = buf732[2]
        assert_size_stride(buf735, (), ())
        buf736 = buf732[3]
        assert_size_stride(buf736, (), ())
        del buf732
        buf737 = buf726; del buf726  # reuse
        # Topologically Sorted Source Nodes: [attn_output_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf733, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_280, (4096, 4096), (1, 4096), 0), out=buf737)
        buf738 = reinterpret_tensor(buf737, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf737  # reuse
        buf739 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf740 = reinterpret_tensor(buf739, (1, s0, 1), (s0, 1, 1), 0); del buf739  # reuse
        buf741 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_305, hidden_states_306, pow_62, variance_61, add_188, rsqrt_61, hidden_states_307, to_127, hidden_states_308], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf738, buf740, buf722, primals_281, buf741, s0, 4096, stream=stream0)
        buf742 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_214], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_282, (4096, 14336), (1, 4096), 0), out=buf742)
        buf743 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_215], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_283, (4096, 14336), (1, 4096), 0), out=buf743)
        buf744 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_30, mul_280], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf742, buf743, buf744, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf745 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_284, (14336, 4096), (1, 14336), 0), out=buf745)
        buf746 = reinterpret_tensor(buf745, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf745  # reuse
        buf747 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf748 = reinterpret_tensor(buf747, (1, s0, 1), (s0, 1, 1), 0); del buf747  # reuse
        buf749 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_309, hidden_states_310, pow_63, variance_62, add_190, rsqrt_62, hidden_states_311, to_129, hidden_states_312], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf746, buf748, buf738, primals_285, buf749, s0, 4096, stream=stream0)
        buf750 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_217], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_286, (4096, 4096), (1, 4096), 0), out=buf750)
        buf751 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [linear_218], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_287, (4096, 1024), (1, 4096), 0), out=buf751)
        buf752 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [linear_219], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_288, (4096, 1024), (1, 4096), 0), out=buf752)
        buf753 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [key_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_2_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf751, buf2, buf753, s0, ps0, triton_poi_fused_clone_2_xnumel, stream=stream0)
        del buf751
        buf754 = empty_strided_cuda((1, 8, 4, s0, 128), (4096*s0, 512*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [value_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf752, buf754, s0, ps0, triton_poi_fused_clone_3_xnumel, stream=stream0)
        del buf752
        buf755 = empty_strided_cuda((1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mul_283, cat_63, mul_284, q_embed_31, query_31], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone]
        triton_poi_fused_add_cat_clone_mul_4_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_mul_4.run(buf750, buf2, buf755, s0, triton_poi_fused_add_cat_clone_mul_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [attn_output_124], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf756 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf755, reinterpret_tensor(buf753, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf754, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf12, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), True, scale=0.08838834764831845)
        buf757 = buf756[0]
        assert_size_stride(buf757, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf758 = buf756[1]
        assert_size_stride(buf758, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
        buf759 = buf756[2]
        assert_size_stride(buf759, (), ())
        buf760 = buf756[3]
        assert_size_stride(buf760, (), ())
        del buf756
        buf761 = buf750; del buf750  # reuse
        # Topologically Sorted Source Nodes: [attn_output_127], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf757, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_289, (4096, 4096), (1, 4096), 0), out=buf761)
        buf762 = reinterpret_tensor(buf761, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf761  # reuse
        buf763 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf764 = reinterpret_tensor(buf763, (1, s0, 1), (s0, 1, 1), 0); del buf763  # reuse
        buf765 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_315, hidden_states_316, pow_64, variance_63, add_194, rsqrt_63, hidden_states_317, to_131, hidden_states_318], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf762, buf764, buf746, primals_290, buf765, s0, 4096, stream=stream0)
        buf766 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_221], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_291, (4096, 14336), (1, 4096), 0), out=buf766)
        buf767 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_222], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_292, (4096, 14336), (1, 4096), 0), out=buf767)
        buf768 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_31, mul_289], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_7.run(buf766, buf767, buf768, triton_poi_fused_mul_silu_7_xnumel, stream=stream0)
        buf769 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [down_proj_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (s0, 14336), (14336, 1), 0), reinterpret_tensor(primals_293, (14336, 4096), (1, 14336), 0), out=buf769)
        buf770 = reinterpret_tensor(buf769, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf769  # reuse
        buf771 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        buf772 = reinterpret_tensor(buf771, (1, s0, 1), (s0, 1, 1), 0); del buf771  # reuse
        buf773 = empty_strided_cuda((1, s0, 4096), (4096*s0, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_319, hidden_states_320, pow_65, variance_64, add_196, rsqrt_64, hidden_states_321, to_133, hidden_states_322], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9.run(buf770, buf772, buf762, primals_294, buf773, s0, 4096, stream=stream0)
        buf774 = empty_strided_cuda((s0, 128256), (128256, 1), torch.float16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf773, (s0, 4096), (4096, 1), 0), reinterpret_tensor(primals_295, (4096, 128256), (1, 4096), 0), out=buf774)
    return (reinterpret_tensor(buf774, (1, s0, 128256), (128256*s0, 128256, 1), 0), primals_2, primals_6, primals_11, primals_15, primals_20, primals_24, primals_29, primals_33, primals_38, primals_42, primals_47, primals_51, primals_56, primals_60, primals_65, primals_69, primals_74, primals_78, primals_83, primals_87, primals_92, primals_96, primals_101, primals_105, primals_110, primals_114, primals_119, primals_123, primals_128, primals_132, primals_137, primals_141, primals_146, primals_150, primals_155, primals_159, primals_164, primals_168, primals_173, primals_177, primals_182, primals_186, primals_191, primals_195, primals_200, primals_204, primals_209, primals_213, primals_218, primals_222, primals_227, primals_231, primals_236, primals_240, primals_245, primals_249, primals_254, primals_258, primals_263, primals_267, primals_272, primals_276, primals_281, primals_285, primals_290, primals_294, buf0, buf2, buf4, reinterpret_tensor(buf5, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf9, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf10, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf11, reinterpret_tensor(buf12, (1, 1, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 8 + s0 + (-1)*(s0 % 8), 1), 0), buf14, buf15, buf16, buf17, buf18, buf20, reinterpret_tensor(buf21, (s0, 4096), (4096, 1), 0), buf22, buf23, reinterpret_tensor(buf24, (s0, 14336), (14336, 1), 0), buf26, buf28, reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf33, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf34, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf35, buf37, buf38, buf39, buf40, buf42, buf44, reinterpret_tensor(buf45, (s0, 4096), (4096, 1), 0), buf46, buf47, reinterpret_tensor(buf48, (s0, 14336), (14336, 1), 0), buf50, buf52, reinterpret_tensor(buf53, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf57, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf58, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf59, buf61, buf62, buf63, buf64, buf66, buf68, reinterpret_tensor(buf69, (s0, 4096), (4096, 1), 0), buf70, buf71, reinterpret_tensor(buf72, (s0, 14336), (14336, 1), 0), buf74, buf76, reinterpret_tensor(buf77, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf81, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf82, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf83, buf85, buf86, buf87, buf88, buf90, buf92, reinterpret_tensor(buf93, (s0, 4096), (4096, 1), 0), buf94, buf95, reinterpret_tensor(buf96, (s0, 14336), (14336, 1), 0), buf98, buf100, reinterpret_tensor(buf101, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf105, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf106, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf107, buf109, buf110, buf111, buf112, buf114, buf116, reinterpret_tensor(buf117, (s0, 4096), (4096, 1), 0), buf118, buf119, reinterpret_tensor(buf120, (s0, 14336), (14336, 1), 0), buf122, buf124, reinterpret_tensor(buf125, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf129, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf130, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf131, buf133, buf134, buf135, buf136, buf138, buf140, reinterpret_tensor(buf141, (s0, 4096), (4096, 1), 0), buf142, buf143, reinterpret_tensor(buf144, (s0, 14336), (14336, 1), 0), buf146, buf148, reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf153, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf154, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf155, buf157, buf158, buf159, buf160, buf162, buf164, reinterpret_tensor(buf165, (s0, 4096), (4096, 1), 0), buf166, buf167, reinterpret_tensor(buf168, (s0, 14336), (14336, 1), 0), buf170, buf172, reinterpret_tensor(buf173, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf177, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf178, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf179, buf181, buf182, buf183, buf184, buf186, buf188, reinterpret_tensor(buf189, (s0, 4096), (4096, 1), 0), buf190, buf191, reinterpret_tensor(buf192, (s0, 14336), (14336, 1), 0), buf194, buf196, reinterpret_tensor(buf197, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf201, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf202, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf203, buf205, buf206, buf207, buf208, buf210, buf212, reinterpret_tensor(buf213, (s0, 4096), (4096, 1), 0), buf214, buf215, reinterpret_tensor(buf216, (s0, 14336), (14336, 1), 0), buf218, buf220, reinterpret_tensor(buf221, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf225, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf226, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf227, buf229, buf230, buf231, buf232, buf234, buf236, reinterpret_tensor(buf237, (s0, 4096), (4096, 1), 0), buf238, buf239, reinterpret_tensor(buf240, (s0, 14336), (14336, 1), 0), buf242, buf244, reinterpret_tensor(buf245, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf249, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf250, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf251, buf253, buf254, buf255, buf256, buf258, buf260, reinterpret_tensor(buf261, (s0, 4096), (4096, 1), 0), buf262, buf263, reinterpret_tensor(buf264, (s0, 14336), (14336, 1), 0), buf266, buf268, reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf273, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf274, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf275, buf277, buf278, buf279, buf280, buf282, buf284, reinterpret_tensor(buf285, (s0, 4096), (4096, 1), 0), buf286, buf287, reinterpret_tensor(buf288, (s0, 14336), (14336, 1), 0), buf290, buf292, reinterpret_tensor(buf293, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf297, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf298, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf299, buf301, buf302, buf303, buf304, buf306, buf308, reinterpret_tensor(buf309, (s0, 4096), (4096, 1), 0), buf310, buf311, reinterpret_tensor(buf312, (s0, 14336), (14336, 1), 0), buf314, buf316, reinterpret_tensor(buf317, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf321, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf322, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf323, buf325, buf326, buf327, buf328, buf330, buf332, reinterpret_tensor(buf333, (s0, 4096), (4096, 1), 0), buf334, buf335, reinterpret_tensor(buf336, (s0, 14336), (14336, 1), 0), buf338, buf340, reinterpret_tensor(buf341, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf345, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf346, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf347, buf349, buf350, buf351, buf352, buf354, buf356, reinterpret_tensor(buf357, (s0, 4096), (4096, 1), 0), buf358, buf359, reinterpret_tensor(buf360, (s0, 14336), (14336, 1), 0), buf362, buf364, reinterpret_tensor(buf365, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf369, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf370, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf371, buf373, buf374, buf375, buf376, buf378, buf380, reinterpret_tensor(buf381, (s0, 4096), (4096, 1), 0), buf382, buf383, reinterpret_tensor(buf384, (s0, 14336), (14336, 1), 0), buf386, buf388, reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf393, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf394, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf395, buf397, buf398, buf399, buf400, buf402, buf404, reinterpret_tensor(buf405, (s0, 4096), (4096, 1), 0), buf406, buf407, reinterpret_tensor(buf408, (s0, 14336), (14336, 1), 0), buf410, buf412, reinterpret_tensor(buf413, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf417, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf418, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf419, buf421, buf422, buf423, buf424, buf426, buf428, reinterpret_tensor(buf429, (s0, 4096), (4096, 1), 0), buf430, buf431, reinterpret_tensor(buf432, (s0, 14336), (14336, 1), 0), buf434, buf436, reinterpret_tensor(buf437, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf441, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf442, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf443, buf445, buf446, buf447, buf448, buf450, buf452, reinterpret_tensor(buf453, (s0, 4096), (4096, 1), 0), buf454, buf455, reinterpret_tensor(buf456, (s0, 14336), (14336, 1), 0), buf458, buf460, reinterpret_tensor(buf461, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf465, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf466, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf467, buf469, buf470, buf471, buf472, buf474, buf476, reinterpret_tensor(buf477, (s0, 4096), (4096, 1), 0), buf478, buf479, reinterpret_tensor(buf480, (s0, 14336), (14336, 1), 0), buf482, buf484, reinterpret_tensor(buf485, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf489, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf490, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf491, buf493, buf494, buf495, buf496, buf498, buf500, reinterpret_tensor(buf501, (s0, 4096), (4096, 1), 0), buf502, buf503, reinterpret_tensor(buf504, (s0, 14336), (14336, 1), 0), buf506, buf508, reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf513, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf514, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf515, buf517, buf518, buf519, buf520, buf522, buf524, reinterpret_tensor(buf525, (s0, 4096), (4096, 1), 0), buf526, buf527, reinterpret_tensor(buf528, (s0, 14336), (14336, 1), 0), buf530, buf532, reinterpret_tensor(buf533, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf537, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf538, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf539, buf541, buf542, buf543, buf544, buf546, buf548, reinterpret_tensor(buf549, (s0, 4096), (4096, 1), 0), buf550, buf551, reinterpret_tensor(buf552, (s0, 14336), (14336, 1), 0), buf554, buf556, reinterpret_tensor(buf557, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf561, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf562, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf563, buf565, buf566, buf567, buf568, buf570, buf572, reinterpret_tensor(buf573, (s0, 4096), (4096, 1), 0), buf574, buf575, reinterpret_tensor(buf576, (s0, 14336), (14336, 1), 0), buf578, buf580, reinterpret_tensor(buf581, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf585, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf586, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf587, buf589, buf590, buf591, buf592, buf594, buf596, reinterpret_tensor(buf597, (s0, 4096), (4096, 1), 0), buf598, buf599, reinterpret_tensor(buf600, (s0, 14336), (14336, 1), 0), buf602, buf604, reinterpret_tensor(buf605, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf609, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf610, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf611, buf613, buf614, buf615, buf616, buf618, buf620, reinterpret_tensor(buf621, (s0, 4096), (4096, 1), 0), buf622, buf623, reinterpret_tensor(buf624, (s0, 14336), (14336, 1), 0), buf626, buf628, reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf633, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf634, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf635, buf637, buf638, buf639, buf640, buf642, buf644, reinterpret_tensor(buf645, (s0, 4096), (4096, 1), 0), buf646, buf647, reinterpret_tensor(buf648, (s0, 14336), (14336, 1), 0), buf650, buf652, reinterpret_tensor(buf653, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf657, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf658, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf659, buf661, buf662, buf663, buf664, buf666, buf668, reinterpret_tensor(buf669, (s0, 4096), (4096, 1), 0), buf670, buf671, reinterpret_tensor(buf672, (s0, 14336), (14336, 1), 0), buf674, buf676, reinterpret_tensor(buf677, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf681, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf682, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf683, buf685, buf686, buf687, buf688, buf690, buf692, reinterpret_tensor(buf693, (s0, 4096), (4096, 1), 0), buf694, buf695, reinterpret_tensor(buf696, (s0, 14336), (14336, 1), 0), buf698, buf700, reinterpret_tensor(buf701, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf705, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf706, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf707, buf709, buf710, buf711, buf712, buf714, buf716, reinterpret_tensor(buf717, (s0, 4096), (4096, 1), 0), buf718, buf719, reinterpret_tensor(buf720, (s0, 14336), (14336, 1), 0), buf722, buf724, reinterpret_tensor(buf725, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf729, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf730, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf731, buf733, buf734, buf735, buf736, buf738, buf740, reinterpret_tensor(buf741, (s0, 4096), (4096, 1), 0), buf742, buf743, reinterpret_tensor(buf744, (s0, 14336), (14336, 1), 0), buf746, buf748, reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0), reinterpret_tensor(buf753, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), reinterpret_tensor(buf754, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1), 0), buf755, buf757, buf758, buf759, buf760, buf762, buf764, reinterpret_tensor(buf765, (s0, 4096), (4096, 1), 0), buf766, buf767, reinterpret_tensor(buf768, (s0, 14336), (14336, 1), 0), buf770, buf772, reinterpret_tensor(buf773, (s0, 4096), (4096, 1), 0), primals_295, primals_293, primals_292, primals_291, primals_289, primals_288, primals_287, primals_286, primals_284, primals_283, primals_282, primals_280, primals_279, primals_278, primals_277, primals_275, primals_274, primals_273, primals_271, primals_270, primals_269, primals_268, primals_266, primals_265, primals_264, primals_262, primals_261, primals_260, primals_259, primals_257, primals_256, primals_255, primals_253, primals_252, primals_251, primals_250, primals_248, primals_247, primals_246, primals_244, primals_243, primals_242, primals_241, primals_239, primals_238, primals_237, primals_235, primals_234, primals_233, primals_232, primals_230, primals_229, primals_228, primals_226, primals_225, primals_224, primals_223, primals_221, primals_220, primals_219, primals_217, primals_216, primals_215, primals_214, primals_212, primals_211, primals_210, primals_208, primals_207, primals_206, primals_205, primals_203, primals_202, primals_201, primals_199, primals_198, primals_197, primals_196, primals_194, primals_193, primals_192, primals_190, primals_189, primals_188, primals_187, primals_185, primals_184, primals_183, primals_181, primals_180, primals_179, primals_178, primals_176, primals_175, primals_174, primals_172, primals_171, primals_170, primals_169, primals_167, primals_166, primals_165, primals_163, primals_162, primals_161, primals_160, primals_158, primals_157, primals_156, primals_154, primals_153, primals_152, primals_151, primals_149, primals_148, primals_147, primals_145, primals_144, primals_143, primals_142, primals_140, primals_139, primals_138, primals_136, primals_135, primals_134, primals_133, primals_131, primals_130, primals_129, primals_127, primals_126, primals_125, primals_124, primals_122, primals_121, primals_120, primals_118, primals_117, primals_116, primals_115, primals_113, primals_112, primals_111, primals_109, primals_108, primals_107, primals_106, primals_104, primals_103, primals_102, primals_100, primals_99, primals_98, primals_97, primals_95, primals_94, primals_93, primals_91, primals_90, primals_89, primals_88, primals_86, primals_85, primals_84, primals_82, primals_81, primals_80, primals_79, primals_77, primals_76, primals_75, primals_73, primals_72, primals_71, primals_70, primals_68, primals_67, primals_66, primals_64, primals_63, primals_62, primals_61, primals_59, primals_58, primals_57, primals_55, primals_54, primals_53, primals_52, primals_50, primals_49, primals_48, primals_46, primals_45, primals_44, primals_43, primals_41, primals_40, primals_39, primals_37, primals_36, primals_35, primals_34, primals_32, primals_31, primals_30, primals_28, primals_27, primals_26, primals_25, primals_23, primals_22, primals_21, primals_19, primals_18, primals_17, primals_16, primals_14, primals_13, primals_12, primals_10, primals_9, primals_8, primals_7, s0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 4
    primals_2 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((128256, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_61 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_63 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_66 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_70 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_72 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_73 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_75 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_76 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_77 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_79 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_80 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_81 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_82 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_83 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_84 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_85 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_86 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_87 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_88 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_89 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_90 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_91 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_93 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_94 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_95 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_96 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_97 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_98 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_99 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_100 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_101 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_102 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_103 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_104 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_105 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_106 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_107 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_108 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_109 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_110 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_111 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_112 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_113 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_114 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_115 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_116 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_117 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_118 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_119 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_120 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_121 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_122 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_124 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_125 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_126 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_127 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_128 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_129 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_130 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_131 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_132 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_133 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_134 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_135 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_136 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_137 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_138 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_139 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_140 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_141 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_142 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_143 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_144 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_145 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_146 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_147 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_148 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_149 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_150 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_151 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_152 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_153 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_154 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_155 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_156 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_157 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_158 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_159 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_160 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_161 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_162 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_163 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_164 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_165 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_166 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_167 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_168 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_169 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_170 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_171 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_172 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_173 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_174 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_175 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_176 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_177 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_178 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_179 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_180 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_181 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_182 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_183 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_184 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_185 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_186 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_187 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_188 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_189 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_190 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_191 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_192 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_193 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_194 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_195 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_196 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_197 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_198 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_199 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_200 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_201 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_202 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_203 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_204 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_205 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_206 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_207 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_208 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_209 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_210 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_211 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_212 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_213 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_214 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_215 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_216 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_217 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_218 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_219 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_220 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_221 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_222 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_223 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_224 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_225 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_226 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_227 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_228 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_229 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_230 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_231 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_232 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_233 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_234 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_235 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_236 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_237 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_238 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_239 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_240 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_241 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_242 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_243 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_244 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_245 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_246 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_247 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_248 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_249 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_250 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_251 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_252 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_253 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_254 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_255 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_256 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_257 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_258 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_259 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_260 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_261 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_262 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_263 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_264 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_265 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_266 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_267 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_268 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_269 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_270 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_271 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_272 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_273 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_274 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_275 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_276 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_277 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_278 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_279 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_280 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_281 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_282 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_283 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_284 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_285 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_286 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_287 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_288 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_289 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_290 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_291 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_292 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    primals_293 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    primals_294 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_295 = rand_strided((128256, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
