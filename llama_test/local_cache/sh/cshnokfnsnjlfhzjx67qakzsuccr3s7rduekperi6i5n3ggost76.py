# AOT ID: ['0_backward']
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


# kernel path: ./local_cache/f7/cf75wfhrx4hlqtiaol5uh7c6pfejp3sdfx5gwjbthqbpdlmpt6mi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_485 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128256, 4096], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_0 = async_compile.triton('triton_poi_fused_embedding_dense_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 525336576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: ./local_cache/dp/cdpmdjxtn76eoilzzn3auquxblbphkbtjfl7kawk25qsvgvr45be.py
# Topologically Sorted Source Nodes: [hidden_states_320, hidden_states_321, to_133], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   hidden_states_320 => convert_element_type_644
#   hidden_states_321 => mul_12445
#   to_133 => convert_element_type_645
# Graph fragment:
#   %convert_element_type_644 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9781, torch.float32), kwargs = {})
#   %mul_12445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_644, %rsqrt_64), kwargs = {})
#   %convert_element_type_645 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12445, torch.float16), kwargs = {})
#   %mul_12480 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_652, %convert_element_type_645), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12480, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_mul_sum_1 = async_compile.triton('triton_red_fused__to_copy_mul_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
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
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: ./local_cache/m2/cm2digt4hcus2fxymbfdumzjafjngblhe2d6qijkeeihhspgjvlp.py
# Topologically Sorted Source Nodes: [hidden_states_320], Original ATen: [aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow, aten.add]
# Source node to ATen node mapping:
#   hidden_states_320 => convert_element_type_644
# Graph fragment:
#   %mul_12479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_652, %primals_294), kwargs = {})
#   %convert_element_type_644 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9781, torch.float32), kwargs = {})
#   %convert_element_type_652 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12479, torch.float32), kwargs = {})
#   %mul_12481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_652, %convert_element_type_644), kwargs = {})
#   %mul_12482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_652, %rsqrt_64), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12481, [2], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_101, 4096), kwargs = {})
#   %pow_67 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_644, 1.0), kwargs = {})
#   %mul_12485 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_67, 2.0), kwargs = {})
#   %mul_12486 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_12485), kwargs = {})
#   %add_9831 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12482, %mul_12486), kwargs = {})
#   %convert_element_type_653 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9831, torch.float16), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_2 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp14 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = -0.5
        tmp17 = tmp8 * tmp16
        tmp18 = tmp14 * tmp14
        tmp19 = tmp18 * tmp14
        tmp20 = tmp17 * tmp19
        tmp21 = 0.000244140625
        tmp22 = tmp20 * tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 2.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp22 * tmp26
        tmp28 = tmp15 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/pi/cpi4fensflvxgbw7pvt5avrpbbatdpbpovn3lahcllivh5a6ucqz.py
# Topologically Sorted Source Nodes: [silu_31], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   silu_31 => convert_element_type_638, convert_element_type_639, mul_12393, sigmoid_31
# Graph fragment:
#   %convert_element_type_638 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_644, torch.float32), kwargs = {})
#   %sigmoid_31 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_638,), kwargs = {})
#   %mul_12393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_638, %sigmoid_31), kwargs = {})
#   %convert_element_type_639 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12393, torch.float16), kwargs = {})
#   %mul_12487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_655, %convert_element_type_639), kwargs = {})
#   %mul_12488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_655, %view_646), kwargs = {})
#   %sigmoid_32 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_644,), kwargs = {})
#   %full_default_68 : [num_users=32] = call_function[target=torch.ops.aten.full.default](args = ([1, %primals_1, 14336], 1), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_3322 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_68, %sigmoid_32), kwargs = {})
#   %mul_12489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_644, %sub_3322), kwargs = {})
#   %add_9832 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_12489, 1), kwargs = {})
#   %mul_12490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_32, %add_9832), kwargs = {})
#   %mul_12491 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12488, %mul_12490), kwargs = {})
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(in_out_ptr0 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/wu/cwurpsdpwva3f5frc5tvzo3fyhfiwqgisf52i24mru6cupi2tm3v.py
# Topologically Sorted Source Nodes: [hidden_states_316], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   hidden_states_316 => convert_element_type_634
# Graph fragment:
#   %add_9833 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_657, %view_659), kwargs = {})
#   %mul_12492 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9833, %primals_290), kwargs = {})
#   %convert_element_type_634 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9716, torch.float32), kwargs = {})
#   %convert_element_type_666 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12492, torch.float32), kwargs = {})
#   %mul_12494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_666, %convert_element_type_634), kwargs = {})
#   %mul_12495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_666, %rsqrt_63), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12494, [2], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_102, 4096), kwargs = {})
#   %pow_69 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_634, 1.0), kwargs = {})
#   %mul_12498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_69, 2.0), kwargs = {})
#   %mul_12499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_12498), kwargs = {})
#   %add_9834 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12495, %mul_12499), kwargs = {})
#   %convert_element_type_667 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9834, torch.float16), kwargs = {})
#   %add_9835 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_653, %convert_element_type_667), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_4 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = -0.5
        tmp22 = tmp10 * tmp21
        tmp23 = tmp19 * tmp19
        tmp24 = tmp23 * tmp19
        tmp25 = tmp22 * tmp24
        tmp26 = 0.000244140625
        tmp27 = tmp25 * tmp26
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 2.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp27 * tmp31
        tmp33 = tmp20 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp12 + tmp34
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/sy/csynl4e2lkownulsdipsvmjq3rt7vqfxqmdgkjqfvmcbsnv7gksn.py
# Topologically Sorted Source Nodes: [hidden_states_316, hidden_states_317, to_131], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   hidden_states_316 => convert_element_type_634
#   hidden_states_317 => mul_12368
#   to_131 => convert_element_type_635
# Graph fragment:
#   %add_9833 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_657, %view_659), kwargs = {})
#   %convert_element_type_634 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9716, torch.float32), kwargs = {})
#   %mul_12368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_634, %rsqrt_63), kwargs = {})
#   %convert_element_type_635 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12368, torch.float16), kwargs = {})
#   %mul_12493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9833, %convert_element_type_635), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12493, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_add_mul_sum_5 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: ./local_cache/tr/ctrdaubodzs5bsh5lodmk3ijrrhhh6yee4zoghuc4e5tatvr6jr7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_100 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_386,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
''', device_str='cuda')


# kernel path: ./local_cache/gj/cgjflhkndk7z5nq6rqc2ds3pfgx4h7sh6dobfuiq7smkh3yy7jlo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_12500 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %unsqueeze_6), kwargs = {})
triton_poi_fused_mul_7 = async_compile.triton('triton_poi_fused_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_7(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x3 = xindex // 128
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 512*x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + 512*x3), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + 512*x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2 + ks0*((x0 % 64))), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tl_math.sin(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp6 * tmp11
    tl.store(out_ptr0 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/33/c33pslorb5xd2k34plkaxwh25sawddifcuabgrgv6gfhofwkut75.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_99 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_381,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
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
''', device_str='cuda')


# kernel path: ./local_cache/kq/ckqknodikymjo64fcqvwkhk3b3gcdw3zuc22qc3lulj7243tdkes.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_98 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_376,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 512*x1), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + 512*x1), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./local_cache/jz/cjzbe2rfzbrzt5nqp4tyglmue73nyj3pprrlwzjs765y5qoh7s2f.py
# Topologically Sorted Source Nodes: [hidden_states_310], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   hidden_states_310 => convert_element_type_624
# Graph fragment:
#   %add_9840 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_668, %view_671), kwargs = {})
#   %add_9841 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9840, %view_674), kwargs = {})
#   %mul_12504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9841, %primals_285), kwargs = {})
#   %convert_element_type_624 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9478, torch.float32), kwargs = {})
#   %convert_element_type_684 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12504, torch.float32), kwargs = {})
#   %mul_12506 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_684, %convert_element_type_624), kwargs = {})
#   %mul_12507 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_684, %rsqrt_62), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12506, [2], True), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_103, 4096), kwargs = {})
#   %pow_71 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_624, 1.0), kwargs = {})
#   %mul_12510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_71, 2.0), kwargs = {})
#   %mul_12511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_12510), kwargs = {})
#   %add_9842 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12507, %mul_12511), kwargs = {})
#   %convert_element_type_685 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9842, torch.float16), kwargs = {})
#   %add_9843 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9835, %convert_element_type_685), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_10 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tmp25 = -0.5
        tmp26 = tmp12 * tmp25
        tmp27 = tmp23 * tmp23
        tmp28 = tmp27 * tmp23
        tmp29 = tmp26 * tmp28
        tmp30 = 0.000244140625
        tmp31 = tmp29 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 2.0
        tmp35 = tmp33 * tmp34
        tmp36 = tmp31 * tmp35
        tmp37 = tmp24 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/oo/cooiaknvye32serylx7xwhx2qwftzzzc42hc3guninlw4pdaroz3.py
# Topologically Sorted Source Nodes: [hidden_states_310, hidden_states_311, to_129], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   hidden_states_310 => convert_element_type_624
#   hidden_states_311 => mul_12059
#   to_129 => convert_element_type_625
# Graph fragment:
#   %add_9840 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_668, %view_671), kwargs = {})
#   %add_9841 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9840, %view_674), kwargs = {})
#   %convert_element_type_624 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9478, torch.float32), kwargs = {})
#   %mul_12059 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_624, %rsqrt_62), kwargs = {})
#   %convert_element_type_625 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12059, torch.float16), kwargs = {})
#   %mul_12505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9841, %convert_element_type_625), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12505, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_add_mul_sum_11 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: ./local_cache/t5/ct5b6awyifp6axgj3o2hglyu23vrueitmssagp2ltfkjebrjoxge.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   hidden_states_5 => add_323
#   hidden_states_6 => convert_element_type_14
# Graph fragment:
#   %add_10205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1339, %view_1341), kwargs = {})
#   %mul_13267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10205, %primals_11), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %convert_element_type_14 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_323, torch.float32), kwargs = {})
#   %convert_element_type_1658 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13267, torch.float32), kwargs = {})
#   %mul_13269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1658, %convert_element_type_14), kwargs = {})
#   %mul_13270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1658, %rsqrt_1), kwargs = {})
#   %sum_190 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13269, [2], True), kwargs = {})
#   %div_63 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_164, 4096), kwargs = {})
#   %pow_193 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 1.0), kwargs = {})
#   %mul_13273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_193, 2.0), kwargs = {})
#   %mul_13274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_63, %mul_13273), kwargs = {})
#   %add_10206 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13270, %mul_13274), kwargs = {})
#   %convert_element_type_1659 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10206, torch.float16), kwargs = {})
#   %add_10207 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10203, %convert_element_type_1659), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_12 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = -0.5
        tmp24 = tmp12 * tmp23
        tmp25 = tmp21 * tmp21
        tmp26 = tmp25 * tmp21
        tmp27 = tmp24 * tmp26
        tmp28 = 0.000244140625
        tmp29 = tmp27 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 2.0
        tmp35 = tmp33 * tmp34
        tmp36 = tmp29 * tmp35
        tmp37 = tmp22 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(in_out_ptr0 + (r0_1 + 4096*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./local_cache/7p/c7pbg3l2pkvqtef7mnprrepoat4qlaxffimkbo2dg7te5trtgllm.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type_4
# Graph fragment:
#   %add_10212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1350, %view_1353), kwargs = {})
#   %add_10213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10212, %view_1356), kwargs = {})
#   %mul_13279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10213, %primals_6), kwargs = {})
#   %convert_element_type_4 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %convert_element_type_1676 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13279, torch.float32), kwargs = {})
#   %mul_13281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1676, %convert_element_type_4), kwargs = {})
#   %mul_13282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1676, %rsqrt), kwargs = {})
#   %sum_194 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13281, [2], True), kwargs = {})
#   %div_64 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_165, 4096), kwargs = {})
#   %pow_195 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 1.0), kwargs = {})
#   %mul_13285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_195, 2.0), kwargs = {})
#   %mul_13286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_64, %mul_13285), kwargs = {})
#   %add_10214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13282, %mul_13286), kwargs = {})
#   %convert_element_type_1677 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10214, torch.float16), kwargs = {})
#   %add_10215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10207, %convert_element_type_1677), kwargs = {})
#   %convert_element_type_1678 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10215, torch.float32), kwargs = {})
#   %full_default_484 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_133, %full_default_484, %convert_element_type_1678), kwargs = {})
#   %full_default_485 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128256, 4096], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_485, [%primals_2], %where_32, True), kwargs = {})
triton_red_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_13 = async_compile.triton('triton_red_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*i64', 'in_ptr6': '*fp16', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_13', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr6 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr4 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.full([1, 1], -1, tl.int64)
        tmp16 = tmp14 == tmp15
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp25 * tmp26
        tmp28 = -0.5
        tmp29 = tmp12 * tmp28
        tmp30 = tmp26 * tmp26
        tmp31 = tmp30 * tmp26
        tmp32 = tmp29 * tmp31
        tmp33 = 0.000244140625
        tmp34 = tmp32 * tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = 2.0
        tmp38 = tmp36 * tmp37
        tmp39 = tmp34 * tmp38
        tmp40 = tmp27 + tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp17 + tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp44 = 0.0
        tmp45 = tl.where(tmp16, tmp44, tmp43)
        tmp46 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp47 = tmp14 + tmp46
        tmp48 = tmp14 < 0
        tmp49 = tl.where(tmp48, tmp47, tmp14)
        tl.device_assert(((0 <= tmp49) & (tmp49 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 128256")
        tl.atomic_add(out_ptr2 + (tl.broadcast_to(r0_1 + 4096*tmp49, [XBLOCK, R0_BLOCK])), tmp45, xmask & r0_mask, sem='relaxed')
''', device_str='cuda')


# kernel path: ./local_cache/gh/cghfemwp5je4gtzbsncc3ktqr4kitawxmdiwtkz7ekfqatqqwhde.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, hidden_states_7, to_7, hidden_states, hidden_states_1, to_5], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type_4
#   hidden_states_1 => mul_93
#   hidden_states_5 => add_323
#   hidden_states_6 => convert_element_type_14
#   hidden_states_7 => mul_402
#   to_5 => convert_element_type_5
#   to_7 => convert_element_type_15
# Graph fragment:
#   %add_10205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1339, %view_1341), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %convert_element_type_14 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_323, torch.float32), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_402, torch.float16), kwargs = {})
#   %mul_13268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10205, %convert_element_type_15), kwargs = {})
#   %sum_189 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13268, [0, 1], True), kwargs = {})
#   %add_10212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1350, %view_1353), kwargs = {})
#   %add_10213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10212, %view_1356), kwargs = {})
#   %convert_element_type_4 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %rsqrt), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.float16), kwargs = {})
#   %mul_13280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10213, %convert_element_type_5), kwargs = {})
#   %sum_193 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13280, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_add_mul_sum_14 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'in_ptr8': '*fp32', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr7 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp3.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp18 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp25, None)
''', device_str='cuda')


# kernel path: ./local_cache/ha/chan4m33wi65uev2xvv4jgjlvl3p2uisjtwxuz4qkgojdpjm52fc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_1679 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.float16), kwargs = {})
triton_poi_fused_embedding_dense_backward_15 = async_compile.triton('triton_poi_fused_embedding_dense_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=60, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6BE828ED5586B6FAE8FF446A7636CA33B7485F906B06FE7C269A644474AB4AF1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 525336576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_6, primals_11, primals_15, primals_20, primals_24, primals_29, primals_33, primals_38, primals_42, primals_47, primals_51, primals_56, primals_60, primals_65, primals_69, primals_74, primals_78, primals_83, primals_87, primals_92, primals_96, primals_101, primals_105, primals_110, primals_114, primals_119, primals_123, primals_128, primals_132, primals_137, primals_141, primals_146, primals_150, primals_155, primals_159, primals_164, primals_168, primals_173, primals_177, primals_182, primals_186, primals_191, primals_195, primals_200, primals_204, primals_209, primals_213, primals_218, primals_222, primals_227, primals_231, primals_236, primals_240, primals_245, primals_249, primals_254, primals_258, primals_263, primals_267, primals_272, primals_276, primals_281, primals_285, primals_290, primals_294, embedding, bmm, rsqrt, view_9, view_18, view_19, clone_4, slice_19, getitem, getitem_1, getitem_2, getitem_3, mm_3, rsqrt_1, view_23, mm_4, mm_5, view_27, add_388, rsqrt_2, view_29, view_38, view_39, clone_7, getitem_4, getitem_5, getitem_6, getitem_7, add_626, rsqrt_3, view_43, mm_11, mm_12, view_47, add_691, rsqrt_4, view_49, view_58, view_59, clone_10, getitem_8, getitem_9, getitem_10, getitem_11, add_929, rsqrt_5, view_63, mm_18, mm_19, view_67, add_994, rsqrt_6, view_69, view_78, view_79, clone_13, getitem_12, getitem_13, getitem_14, getitem_15, add_1232, rsqrt_7, view_83, mm_25, mm_26, view_87, add_1297, rsqrt_8, view_89, view_98, view_99, clone_16, getitem_16, getitem_17, getitem_18, getitem_19, add_1535, rsqrt_9, view_103, mm_32, mm_33, view_107, add_1600, rsqrt_10, view_109, view_118, view_119, clone_19, getitem_20, getitem_21, getitem_22, getitem_23, add_1838, rsqrt_11, view_123, mm_39, mm_40, view_127, add_1903, rsqrt_12, view_129, view_138, view_139, clone_22, getitem_24, getitem_25, getitem_26, getitem_27, add_2141, rsqrt_13, view_143, mm_46, mm_47, view_147, add_2206, rsqrt_14, view_149, view_158, view_159, clone_25, getitem_28, getitem_29, getitem_30, getitem_31, add_2444, rsqrt_15, view_163, mm_53, mm_54, view_167, add_2509, rsqrt_16, view_169, view_178, view_179, clone_28, getitem_32, getitem_33, getitem_34, getitem_35, add_2747, rsqrt_17, view_183, mm_60, mm_61, view_187, add_2812, rsqrt_18, view_189, view_198, view_199, clone_31, getitem_36, getitem_37, getitem_38, getitem_39, add_3050, rsqrt_19, view_203, mm_67, mm_68, view_207, add_3115, rsqrt_20, view_209, view_218, view_219, clone_34, getitem_40, getitem_41, getitem_42, getitem_43, add_3353, rsqrt_21, view_223, mm_74, mm_75, view_227, add_3418, rsqrt_22, view_229, view_238, view_239, clone_37, getitem_44, getitem_45, getitem_46, getitem_47, add_3656, rsqrt_23, view_243, mm_81, mm_82, view_247, add_3721, rsqrt_24, view_249, view_258, view_259, clone_40, getitem_48, getitem_49, getitem_50, getitem_51, add_3959, rsqrt_25, view_263, mm_88, mm_89, view_267, add_4024, rsqrt_26, view_269, view_278, view_279, clone_43, getitem_52, getitem_53, getitem_54, getitem_55, add_4262, rsqrt_27, view_283, mm_95, mm_96, view_287, add_4327, rsqrt_28, view_289, view_298, view_299, clone_46, getitem_56, getitem_57, getitem_58, getitem_59, add_4565, rsqrt_29, view_303, mm_102, mm_103, view_307, add_4630, rsqrt_30, view_309, view_318, view_319, clone_49, getitem_60, getitem_61, getitem_62, getitem_63, add_4868, rsqrt_31, view_323, mm_109, mm_110, view_327, add_4933, rsqrt_32, view_329, view_338, view_339, clone_52, getitem_64, getitem_65, getitem_66, getitem_67, add_5171, rsqrt_33, view_343, mm_116, mm_117, view_347, add_5236, rsqrt_34, view_349, view_358, view_359, clone_55, getitem_68, getitem_69, getitem_70, getitem_71, add_5474, rsqrt_35, view_363, mm_123, mm_124, view_367, add_5539, rsqrt_36, view_369, view_378, view_379, clone_58, getitem_72, getitem_73, getitem_74, getitem_75, add_5777, rsqrt_37, view_383, mm_130, mm_131, view_387, add_5842, rsqrt_38, view_389, view_398, view_399, clone_61, getitem_76, getitem_77, getitem_78, getitem_79, add_6080, rsqrt_39, view_403, mm_137, mm_138, view_407, add_6145, rsqrt_40, view_409, view_418, view_419, clone_64, getitem_80, getitem_81, getitem_82, getitem_83, add_6383, rsqrt_41, view_423, mm_144, mm_145, view_427, add_6448, rsqrt_42, view_429, view_438, view_439, clone_67, getitem_84, getitem_85, getitem_86, getitem_87, add_6686, rsqrt_43, view_443, mm_151, mm_152, view_447, add_6751, rsqrt_44, view_449, view_458, view_459, clone_70, getitem_88, getitem_89, getitem_90, getitem_91, add_6989, rsqrt_45, view_463, mm_158, mm_159, view_467, add_7054, rsqrt_46, view_469, view_478, view_479, clone_73, getitem_92, getitem_93, getitem_94, getitem_95, add_7292, rsqrt_47, view_483, mm_165, mm_166, view_487, add_7357, rsqrt_48, view_489, view_498, view_499, clone_76, getitem_96, getitem_97, getitem_98, getitem_99, add_7595, rsqrt_49, view_503, mm_172, mm_173, view_507, add_7660, rsqrt_50, view_509, view_518, view_519, clone_79, getitem_100, getitem_101, getitem_102, getitem_103, add_7898, rsqrt_51, view_523, mm_179, mm_180, view_527, add_7963, rsqrt_52, view_529, view_538, view_539, clone_82, getitem_104, getitem_105, getitem_106, getitem_107, add_8201, rsqrt_53, view_543, mm_186, mm_187, view_547, add_8266, rsqrt_54, view_549, view_558, view_559, clone_85, getitem_108, getitem_109, getitem_110, getitem_111, add_8504, rsqrt_55, view_563, mm_193, mm_194, view_567, add_8569, rsqrt_56, view_569, view_578, view_579, clone_88, getitem_112, getitem_113, getitem_114, getitem_115, add_8807, rsqrt_57, view_583, mm_200, mm_201, view_587, add_8872, rsqrt_58, view_589, view_598, view_599, clone_91, getitem_116, getitem_117, getitem_118, getitem_119, add_9110, rsqrt_59, view_603, mm_207, mm_208, view_607, add_9175, rsqrt_60, view_609, view_618, view_619, clone_94, getitem_120, getitem_121, getitem_122, getitem_123, add_9413, rsqrt_61, view_623, mm_214, mm_215, view_627, add_9478, rsqrt_62, view_629, view_638, view_639, clone_97, getitem_124, getitem_125, getitem_126, getitem_127, add_9716, rsqrt_63, view_643, mm_221, mm_222, view_647, add_9781, rsqrt_64, view_649, permute_356, permute_360, permute_364, permute_369, permute_373, permute_379, permute_384, permute_389, permute_393, permute_397, permute_402, permute_406, permute_412, permute_417, permute_422, permute_426, permute_430, permute_435, permute_439, permute_445, permute_450, permute_455, permute_459, permute_463, permute_468, permute_472, permute_478, permute_483, permute_488, permute_492, permute_496, permute_501, permute_505, permute_511, permute_516, permute_521, permute_525, permute_529, permute_534, permute_538, permute_544, permute_549, permute_554, permute_558, permute_562, permute_567, permute_571, permute_577, permute_582, permute_587, permute_591, permute_595, permute_600, permute_604, permute_610, permute_615, permute_620, permute_624, permute_628, permute_633, permute_637, permute_643, permute_648, permute_653, permute_657, permute_661, permute_666, permute_670, permute_676, permute_681, permute_686, permute_690, permute_694, permute_699, permute_703, permute_709, permute_714, permute_719, permute_723, permute_727, permute_732, permute_736, permute_742, permute_747, permute_752, permute_756, permute_760, permute_765, permute_769, permute_775, permute_780, permute_785, permute_789, permute_793, permute_798, permute_802, permute_808, permute_813, permute_818, permute_822, permute_826, permute_831, permute_835, permute_841, permute_846, permute_851, permute_855, permute_859, permute_864, permute_868, permute_874, permute_879, permute_884, permute_888, permute_892, permute_897, permute_901, permute_907, permute_912, permute_917, permute_921, permute_925, permute_930, permute_934, permute_940, permute_945, permute_950, permute_954, permute_958, permute_963, permute_967, permute_973, permute_978, permute_983, permute_987, permute_991, permute_996, permute_1000, permute_1006, permute_1011, permute_1016, permute_1020, permute_1024, permute_1029, permute_1033, permute_1039, permute_1044, permute_1049, permute_1053, permute_1057, permute_1062, permute_1066, permute_1072, permute_1077, permute_1082, permute_1086, permute_1090, permute_1095, permute_1099, permute_1105, permute_1110, permute_1115, permute_1119, permute_1123, permute_1128, permute_1132, permute_1138, permute_1143, permute_1148, permute_1152, permute_1156, permute_1161, permute_1165, permute_1171, permute_1176, permute_1181, permute_1185, permute_1189, permute_1194, permute_1198, permute_1204, permute_1209, permute_1214, permute_1218, permute_1222, permute_1227, permute_1231, permute_1237, permute_1242, permute_1247, permute_1251, permute_1255, permute_1260, permute_1264, permute_1270, permute_1275, permute_1280, permute_1284, permute_1288, permute_1293, permute_1297, permute_1303, permute_1308, permute_1313, permute_1317, permute_1321, permute_1326, permute_1330, permute_1336, permute_1341, permute_1346, permute_1350, permute_1354, permute_1359, permute_1363, permute_1369, permute_1374, permute_1379, permute_1383, permute_1387, permute_1392, permute_1396, permute_1402, permute_1407, permute_1412, tangents_1 = args
    args.clear()
    s0 = primals_1
    assert_size_stride(primals_2, (1, s0), (s0, 1))
    assert_size_stride(primals_6, (4096, ), (1, ))
    assert_size_stride(primals_11, (4096, ), (1, ))
    assert_size_stride(primals_15, (4096, ), (1, ))
    assert_size_stride(primals_20, (4096, ), (1, ))
    assert_size_stride(primals_24, (4096, ), (1, ))
    assert_size_stride(primals_29, (4096, ), (1, ))
    assert_size_stride(primals_33, (4096, ), (1, ))
    assert_size_stride(primals_38, (4096, ), (1, ))
    assert_size_stride(primals_42, (4096, ), (1, ))
    assert_size_stride(primals_47, (4096, ), (1, ))
    assert_size_stride(primals_51, (4096, ), (1, ))
    assert_size_stride(primals_56, (4096, ), (1, ))
    assert_size_stride(primals_60, (4096, ), (1, ))
    assert_size_stride(primals_65, (4096, ), (1, ))
    assert_size_stride(primals_69, (4096, ), (1, ))
    assert_size_stride(primals_74, (4096, ), (1, ))
    assert_size_stride(primals_78, (4096, ), (1, ))
    assert_size_stride(primals_83, (4096, ), (1, ))
    assert_size_stride(primals_87, (4096, ), (1, ))
    assert_size_stride(primals_92, (4096, ), (1, ))
    assert_size_stride(primals_96, (4096, ), (1, ))
    assert_size_stride(primals_101, (4096, ), (1, ))
    assert_size_stride(primals_105, (4096, ), (1, ))
    assert_size_stride(primals_110, (4096, ), (1, ))
    assert_size_stride(primals_114, (4096, ), (1, ))
    assert_size_stride(primals_119, (4096, ), (1, ))
    assert_size_stride(primals_123, (4096, ), (1, ))
    assert_size_stride(primals_128, (4096, ), (1, ))
    assert_size_stride(primals_132, (4096, ), (1, ))
    assert_size_stride(primals_137, (4096, ), (1, ))
    assert_size_stride(primals_141, (4096, ), (1, ))
    assert_size_stride(primals_146, (4096, ), (1, ))
    assert_size_stride(primals_150, (4096, ), (1, ))
    assert_size_stride(primals_155, (4096, ), (1, ))
    assert_size_stride(primals_159, (4096, ), (1, ))
    assert_size_stride(primals_164, (4096, ), (1, ))
    assert_size_stride(primals_168, (4096, ), (1, ))
    assert_size_stride(primals_173, (4096, ), (1, ))
    assert_size_stride(primals_177, (4096, ), (1, ))
    assert_size_stride(primals_182, (4096, ), (1, ))
    assert_size_stride(primals_186, (4096, ), (1, ))
    assert_size_stride(primals_191, (4096, ), (1, ))
    assert_size_stride(primals_195, (4096, ), (1, ))
    assert_size_stride(primals_200, (4096, ), (1, ))
    assert_size_stride(primals_204, (4096, ), (1, ))
    assert_size_stride(primals_209, (4096, ), (1, ))
    assert_size_stride(primals_213, (4096, ), (1, ))
    assert_size_stride(primals_218, (4096, ), (1, ))
    assert_size_stride(primals_222, (4096, ), (1, ))
    assert_size_stride(primals_227, (4096, ), (1, ))
    assert_size_stride(primals_231, (4096, ), (1, ))
    assert_size_stride(primals_236, (4096, ), (1, ))
    assert_size_stride(primals_240, (4096, ), (1, ))
    assert_size_stride(primals_245, (4096, ), (1, ))
    assert_size_stride(primals_249, (4096, ), (1, ))
    assert_size_stride(primals_254, (4096, ), (1, ))
    assert_size_stride(primals_258, (4096, ), (1, ))
    assert_size_stride(primals_263, (4096, ), (1, ))
    assert_size_stride(primals_267, (4096, ), (1, ))
    assert_size_stride(primals_272, (4096, ), (1, ))
    assert_size_stride(primals_276, (4096, ), (1, ))
    assert_size_stride(primals_281, (4096, ), (1, ))
    assert_size_stride(primals_285, (4096, ), (1, ))
    assert_size_stride(primals_290, (4096, ), (1, ))
    assert_size_stride(primals_294, (4096, ), (1, ))
    assert_size_stride(embedding, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(bmm, (1, 64, s0), (64*s0, s0, 1))
    assert_size_stride(rsqrt, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_9, (s0, 4096), (4096, 1))
    assert_size_stride(view_18, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_19, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_4, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(slice_19, (1, 1, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 8 + s0 + (-1)*(s0 % 8), 1))
    assert_size_stride(getitem, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_1, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_2, (), ())
    assert_size_stride(getitem_3, (), ())
    assert_size_stride(mm_3, (s0, 4096), (4096, 1))
    assert_size_stride(rsqrt_1, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_23, (s0, 4096), (4096, 1))
    assert_size_stride(mm_4, (s0, 14336), (14336, 1))
    assert_size_stride(mm_5, (s0, 14336), (14336, 1))
    assert_size_stride(view_27, (s0, 14336), (14336, 1))
    assert_size_stride(add_388, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_2, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_29, (s0, 4096), (4096, 1))
    assert_size_stride(view_38, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_39, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_7, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_4, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_5, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(add_626, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_3, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_43, (s0, 4096), (4096, 1))
    assert_size_stride(mm_11, (s0, 14336), (14336, 1))
    assert_size_stride(mm_12, (s0, 14336), (14336, 1))
    assert_size_stride(view_47, (s0, 14336), (14336, 1))
    assert_size_stride(add_691, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_4, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_49, (s0, 4096), (4096, 1))
    assert_size_stride(view_58, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_59, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_10, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_8, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_9, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_10, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(add_929, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_5, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_63, (s0, 4096), (4096, 1))
    assert_size_stride(mm_18, (s0, 14336), (14336, 1))
    assert_size_stride(mm_19, (s0, 14336), (14336, 1))
    assert_size_stride(view_67, (s0, 14336), (14336, 1))
    assert_size_stride(add_994, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_6, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_69, (s0, 4096), (4096, 1))
    assert_size_stride(view_78, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_79, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_13, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_12, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_13, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_14, (), ())
    assert_size_stride(getitem_15, (), ())
    assert_size_stride(add_1232, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_7, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_83, (s0, 4096), (4096, 1))
    assert_size_stride(mm_25, (s0, 14336), (14336, 1))
    assert_size_stride(mm_26, (s0, 14336), (14336, 1))
    assert_size_stride(view_87, (s0, 14336), (14336, 1))
    assert_size_stride(add_1297, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_8, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_89, (s0, 4096), (4096, 1))
    assert_size_stride(view_98, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_99, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_16, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_16, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_17, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(add_1535, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_9, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_103, (s0, 4096), (4096, 1))
    assert_size_stride(mm_32, (s0, 14336), (14336, 1))
    assert_size_stride(mm_33, (s0, 14336), (14336, 1))
    assert_size_stride(view_107, (s0, 14336), (14336, 1))
    assert_size_stride(add_1600, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_10, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_109, (s0, 4096), (4096, 1))
    assert_size_stride(view_118, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_119, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_19, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_20, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_21, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_22, (), ())
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(add_1838, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_11, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_123, (s0, 4096), (4096, 1))
    assert_size_stride(mm_39, (s0, 14336), (14336, 1))
    assert_size_stride(mm_40, (s0, 14336), (14336, 1))
    assert_size_stride(view_127, (s0, 14336), (14336, 1))
    assert_size_stride(add_1903, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_12, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_129, (s0, 4096), (4096, 1))
    assert_size_stride(view_138, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_139, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_22, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_24, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_25, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_26, (), ())
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(add_2141, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_13, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_143, (s0, 4096), (4096, 1))
    assert_size_stride(mm_46, (s0, 14336), (14336, 1))
    assert_size_stride(mm_47, (s0, 14336), (14336, 1))
    assert_size_stride(view_147, (s0, 14336), (14336, 1))
    assert_size_stride(add_2206, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_14, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_149, (s0, 4096), (4096, 1))
    assert_size_stride(view_158, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_159, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_25, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_28, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_29, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(getitem_31, (), ())
    assert_size_stride(add_2444, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_15, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_163, (s0, 4096), (4096, 1))
    assert_size_stride(mm_53, (s0, 14336), (14336, 1))
    assert_size_stride(mm_54, (s0, 14336), (14336, 1))
    assert_size_stride(view_167, (s0, 14336), (14336, 1))
    assert_size_stride(add_2509, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_16, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_169, (s0, 4096), (4096, 1))
    assert_size_stride(view_178, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_179, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_28, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_32, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_33, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_34, (), ())
    assert_size_stride(getitem_35, (), ())
    assert_size_stride(add_2747, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_17, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_183, (s0, 4096), (4096, 1))
    assert_size_stride(mm_60, (s0, 14336), (14336, 1))
    assert_size_stride(mm_61, (s0, 14336), (14336, 1))
    assert_size_stride(view_187, (s0, 14336), (14336, 1))
    assert_size_stride(add_2812, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_18, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_189, (s0, 4096), (4096, 1))
    assert_size_stride(view_198, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_199, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_31, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_36, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_37, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_38, (), ())
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(add_3050, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_19, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_203, (s0, 4096), (4096, 1))
    assert_size_stride(mm_67, (s0, 14336), (14336, 1))
    assert_size_stride(mm_68, (s0, 14336), (14336, 1))
    assert_size_stride(view_207, (s0, 14336), (14336, 1))
    assert_size_stride(add_3115, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_20, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_209, (s0, 4096), (4096, 1))
    assert_size_stride(view_218, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_219, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_34, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_40, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_41, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_42, (), ())
    assert_size_stride(getitem_43, (), ())
    assert_size_stride(add_3353, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_21, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_223, (s0, 4096), (4096, 1))
    assert_size_stride(mm_74, (s0, 14336), (14336, 1))
    assert_size_stride(mm_75, (s0, 14336), (14336, 1))
    assert_size_stride(view_227, (s0, 14336), (14336, 1))
    assert_size_stride(add_3418, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_22, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_229, (s0, 4096), (4096, 1))
    assert_size_stride(view_238, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_239, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_37, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_44, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_45, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_46, (), ())
    assert_size_stride(getitem_47, (), ())
    assert_size_stride(add_3656, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_23, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_243, (s0, 4096), (4096, 1))
    assert_size_stride(mm_81, (s0, 14336), (14336, 1))
    assert_size_stride(mm_82, (s0, 14336), (14336, 1))
    assert_size_stride(view_247, (s0, 14336), (14336, 1))
    assert_size_stride(add_3721, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_24, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_249, (s0, 4096), (4096, 1))
    assert_size_stride(view_258, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_259, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_40, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_48, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_49, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_50, (), ())
    assert_size_stride(getitem_51, (), ())
    assert_size_stride(add_3959, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_25, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_263, (s0, 4096), (4096, 1))
    assert_size_stride(mm_88, (s0, 14336), (14336, 1))
    assert_size_stride(mm_89, (s0, 14336), (14336, 1))
    assert_size_stride(view_267, (s0, 14336), (14336, 1))
    assert_size_stride(add_4024, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_26, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_269, (s0, 4096), (4096, 1))
    assert_size_stride(view_278, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_279, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_43, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_52, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_53, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_54, (), ())
    assert_size_stride(getitem_55, (), ())
    assert_size_stride(add_4262, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_27, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_283, (s0, 4096), (4096, 1))
    assert_size_stride(mm_95, (s0, 14336), (14336, 1))
    assert_size_stride(mm_96, (s0, 14336), (14336, 1))
    assert_size_stride(view_287, (s0, 14336), (14336, 1))
    assert_size_stride(add_4327, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_28, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_289, (s0, 4096), (4096, 1))
    assert_size_stride(view_298, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_299, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_46, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_56, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_57, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_58, (), ())
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(add_4565, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_29, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_303, (s0, 4096), (4096, 1))
    assert_size_stride(mm_102, (s0, 14336), (14336, 1))
    assert_size_stride(mm_103, (s0, 14336), (14336, 1))
    assert_size_stride(view_307, (s0, 14336), (14336, 1))
    assert_size_stride(add_4630, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_30, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_309, (s0, 4096), (4096, 1))
    assert_size_stride(view_318, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_319, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_49, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_60, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_61, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(add_4868, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_31, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_323, (s0, 4096), (4096, 1))
    assert_size_stride(mm_109, (s0, 14336), (14336, 1))
    assert_size_stride(mm_110, (s0, 14336), (14336, 1))
    assert_size_stride(view_327, (s0, 14336), (14336, 1))
    assert_size_stride(add_4933, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_32, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_329, (s0, 4096), (4096, 1))
    assert_size_stride(view_338, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_339, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_52, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_64, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_65, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_66, (), ())
    assert_size_stride(getitem_67, (), ())
    assert_size_stride(add_5171, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_33, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_343, (s0, 4096), (4096, 1))
    assert_size_stride(mm_116, (s0, 14336), (14336, 1))
    assert_size_stride(mm_117, (s0, 14336), (14336, 1))
    assert_size_stride(view_347, (s0, 14336), (14336, 1))
    assert_size_stride(add_5236, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_34, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_349, (s0, 4096), (4096, 1))
    assert_size_stride(view_358, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_359, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_55, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_68, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_69, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_70, (), ())
    assert_size_stride(getitem_71, (), ())
    assert_size_stride(add_5474, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_35, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_363, (s0, 4096), (4096, 1))
    assert_size_stride(mm_123, (s0, 14336), (14336, 1))
    assert_size_stride(mm_124, (s0, 14336), (14336, 1))
    assert_size_stride(view_367, (s0, 14336), (14336, 1))
    assert_size_stride(add_5539, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_36, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_369, (s0, 4096), (4096, 1))
    assert_size_stride(view_378, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_379, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_58, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_72, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_73, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(getitem_75, (), ())
    assert_size_stride(add_5777, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_37, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_383, (s0, 4096), (4096, 1))
    assert_size_stride(mm_130, (s0, 14336), (14336, 1))
    assert_size_stride(mm_131, (s0, 14336), (14336, 1))
    assert_size_stride(view_387, (s0, 14336), (14336, 1))
    assert_size_stride(add_5842, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_38, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_389, (s0, 4096), (4096, 1))
    assert_size_stride(view_398, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_399, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_61, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_76, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_77, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_78, (), ())
    assert_size_stride(getitem_79, (), ())
    assert_size_stride(add_6080, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_39, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_403, (s0, 4096), (4096, 1))
    assert_size_stride(mm_137, (s0, 14336), (14336, 1))
    assert_size_stride(mm_138, (s0, 14336), (14336, 1))
    assert_size_stride(view_407, (s0, 14336), (14336, 1))
    assert_size_stride(add_6145, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_40, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_409, (s0, 4096), (4096, 1))
    assert_size_stride(view_418, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_419, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_64, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_80, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_81, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_82, (), ())
    assert_size_stride(getitem_83, (), ())
    assert_size_stride(add_6383, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_41, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_423, (s0, 4096), (4096, 1))
    assert_size_stride(mm_144, (s0, 14336), (14336, 1))
    assert_size_stride(mm_145, (s0, 14336), (14336, 1))
    assert_size_stride(view_427, (s0, 14336), (14336, 1))
    assert_size_stride(add_6448, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_42, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_429, (s0, 4096), (4096, 1))
    assert_size_stride(view_438, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_439, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_67, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_84, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_85, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_86, (), ())
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(add_6686, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_43, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_443, (s0, 4096), (4096, 1))
    assert_size_stride(mm_151, (s0, 14336), (14336, 1))
    assert_size_stride(mm_152, (s0, 14336), (14336, 1))
    assert_size_stride(view_447, (s0, 14336), (14336, 1))
    assert_size_stride(add_6751, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_44, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_449, (s0, 4096), (4096, 1))
    assert_size_stride(view_458, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_459, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_70, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_88, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_89, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_90, (), ())
    assert_size_stride(getitem_91, (), ())
    assert_size_stride(add_6989, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_45, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_463, (s0, 4096), (4096, 1))
    assert_size_stride(mm_158, (s0, 14336), (14336, 1))
    assert_size_stride(mm_159, (s0, 14336), (14336, 1))
    assert_size_stride(view_467, (s0, 14336), (14336, 1))
    assert_size_stride(add_7054, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_46, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_469, (s0, 4096), (4096, 1))
    assert_size_stride(view_478, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_479, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_73, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_92, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_93, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_94, (), ())
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(add_7292, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_47, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_483, (s0, 4096), (4096, 1))
    assert_size_stride(mm_165, (s0, 14336), (14336, 1))
    assert_size_stride(mm_166, (s0, 14336), (14336, 1))
    assert_size_stride(view_487, (s0, 14336), (14336, 1))
    assert_size_stride(add_7357, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_48, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_489, (s0, 4096), (4096, 1))
    assert_size_stride(view_498, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_499, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_76, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_96, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_97, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_98, (), ())
    assert_size_stride(getitem_99, (), ())
    assert_size_stride(add_7595, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_49, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_503, (s0, 4096), (4096, 1))
    assert_size_stride(mm_172, (s0, 14336), (14336, 1))
    assert_size_stride(mm_173, (s0, 14336), (14336, 1))
    assert_size_stride(view_507, (s0, 14336), (14336, 1))
    assert_size_stride(add_7660, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_50, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_509, (s0, 4096), (4096, 1))
    assert_size_stride(view_518, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_519, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_79, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_100, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_101, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_102, (), ())
    assert_size_stride(getitem_103, (), ())
    assert_size_stride(add_7898, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_51, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_523, (s0, 4096), (4096, 1))
    assert_size_stride(mm_179, (s0, 14336), (14336, 1))
    assert_size_stride(mm_180, (s0, 14336), (14336, 1))
    assert_size_stride(view_527, (s0, 14336), (14336, 1))
    assert_size_stride(add_7963, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_52, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_529, (s0, 4096), (4096, 1))
    assert_size_stride(view_538, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_539, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_82, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_104, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_105, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_106, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(add_8201, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_53, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_543, (s0, 4096), (4096, 1))
    assert_size_stride(mm_186, (s0, 14336), (14336, 1))
    assert_size_stride(mm_187, (s0, 14336), (14336, 1))
    assert_size_stride(view_547, (s0, 14336), (14336, 1))
    assert_size_stride(add_8266, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_54, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_549, (s0, 4096), (4096, 1))
    assert_size_stride(view_558, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_559, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_85, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_108, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_109, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_110, (), ())
    assert_size_stride(getitem_111, (), ())
    assert_size_stride(add_8504, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_55, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_563, (s0, 4096), (4096, 1))
    assert_size_stride(mm_193, (s0, 14336), (14336, 1))
    assert_size_stride(mm_194, (s0, 14336), (14336, 1))
    assert_size_stride(view_567, (s0, 14336), (14336, 1))
    assert_size_stride(add_8569, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_56, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_569, (s0, 4096), (4096, 1))
    assert_size_stride(view_578, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_579, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_88, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_112, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_113, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_114, (), ())
    assert_size_stride(getitem_115, (), ())
    assert_size_stride(add_8807, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_57, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_583, (s0, 4096), (4096, 1))
    assert_size_stride(mm_200, (s0, 14336), (14336, 1))
    assert_size_stride(mm_201, (s0, 14336), (14336, 1))
    assert_size_stride(view_587, (s0, 14336), (14336, 1))
    assert_size_stride(add_8872, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_58, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_589, (s0, 4096), (4096, 1))
    assert_size_stride(view_598, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_599, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_91, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_116, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_117, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_118, (), ())
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(add_9110, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_59, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_603, (s0, 4096), (4096, 1))
    assert_size_stride(mm_207, (s0, 14336), (14336, 1))
    assert_size_stride(mm_208, (s0, 14336), (14336, 1))
    assert_size_stride(view_607, (s0, 14336), (14336, 1))
    assert_size_stride(add_9175, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_60, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_609, (s0, 4096), (4096, 1))
    assert_size_stride(view_618, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_619, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_94, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_120, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_121, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_122, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(add_9413, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_61, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_623, (s0, 4096), (4096, 1))
    assert_size_stride(mm_214, (s0, 14336), (14336, 1))
    assert_size_stride(mm_215, (s0, 14336), (14336, 1))
    assert_size_stride(view_627, (s0, 14336), (14336, 1))
    assert_size_stride(add_9478, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_62, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_629, (s0, 4096), (4096, 1))
    assert_size_stride(view_638, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(view_639, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(clone_97, (1, 32, s0, 128), (4096*s0, 128*s0, 128, 1))
    assert_size_stride(getitem_124, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
    assert_size_stride(getitem_125, (1, 32, 32*math.ceil(s0 / 32)), (1024*math.ceil(s0 / 32), 32*math.ceil(s0 / 32), 1))
    assert_size_stride(getitem_126, (), ())
    assert_size_stride(getitem_127, (), ())
    assert_size_stride(add_9716, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_63, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_643, (s0, 4096), (4096, 1))
    assert_size_stride(mm_221, (s0, 14336), (14336, 1))
    assert_size_stride(mm_222, (s0, 14336), (14336, 1))
    assert_size_stride(view_647, (s0, 14336), (14336, 1))
    assert_size_stride(add_9781, (1, s0, 4096), (4096*s0, 4096, 1))
    assert_size_stride(rsqrt_64, (1, s0, 1), (s0, 1, 1))
    assert_size_stride(view_649, (s0, 4096), (4096, 1))
    assert_size_stride(permute_356, (128256, 4096), (4096, 1))
    assert_size_stride(permute_360, (4096, 14336), (14336, 1))
    assert_size_stride(permute_364, (14336, 4096), (4096, 1))
    assert_size_stride(permute_369, (14336, 4096), (4096, 1))
    assert_size_stride(permute_373, (4096, 4096), (4096, 1))
    assert_size_stride(permute_379, (1024, 4096), (4096, 1))
    assert_size_stride(permute_384, (1024, 4096), (4096, 1))
    assert_size_stride(permute_389, (4096, 4096), (4096, 1))
    assert_size_stride(permute_393, (4096, 14336), (14336, 1))
    assert_size_stride(permute_397, (14336, 4096), (4096, 1))
    assert_size_stride(permute_402, (14336, 4096), (4096, 1))
    assert_size_stride(permute_406, (4096, 4096), (4096, 1))
    assert_size_stride(permute_412, (1024, 4096), (4096, 1))
    assert_size_stride(permute_417, (1024, 4096), (4096, 1))
    assert_size_stride(permute_422, (4096, 4096), (4096, 1))
    assert_size_stride(permute_426, (4096, 14336), (14336, 1))
    assert_size_stride(permute_430, (14336, 4096), (4096, 1))
    assert_size_stride(permute_435, (14336, 4096), (4096, 1))
    assert_size_stride(permute_439, (4096, 4096), (4096, 1))
    assert_size_stride(permute_445, (1024, 4096), (4096, 1))
    assert_size_stride(permute_450, (1024, 4096), (4096, 1))
    assert_size_stride(permute_455, (4096, 4096), (4096, 1))
    assert_size_stride(permute_459, (4096, 14336), (14336, 1))
    assert_size_stride(permute_463, (14336, 4096), (4096, 1))
    assert_size_stride(permute_468, (14336, 4096), (4096, 1))
    assert_size_stride(permute_472, (4096, 4096), (4096, 1))
    assert_size_stride(permute_478, (1024, 4096), (4096, 1))
    assert_size_stride(permute_483, (1024, 4096), (4096, 1))
    assert_size_stride(permute_488, (4096, 4096), (4096, 1))
    assert_size_stride(permute_492, (4096, 14336), (14336, 1))
    assert_size_stride(permute_496, (14336, 4096), (4096, 1))
    assert_size_stride(permute_501, (14336, 4096), (4096, 1))
    assert_size_stride(permute_505, (4096, 4096), (4096, 1))
    assert_size_stride(permute_511, (1024, 4096), (4096, 1))
    assert_size_stride(permute_516, (1024, 4096), (4096, 1))
    assert_size_stride(permute_521, (4096, 4096), (4096, 1))
    assert_size_stride(permute_525, (4096, 14336), (14336, 1))
    assert_size_stride(permute_529, (14336, 4096), (4096, 1))
    assert_size_stride(permute_534, (14336, 4096), (4096, 1))
    assert_size_stride(permute_538, (4096, 4096), (4096, 1))
    assert_size_stride(permute_544, (1024, 4096), (4096, 1))
    assert_size_stride(permute_549, (1024, 4096), (4096, 1))
    assert_size_stride(permute_554, (4096, 4096), (4096, 1))
    assert_size_stride(permute_558, (4096, 14336), (14336, 1))
    assert_size_stride(permute_562, (14336, 4096), (4096, 1))
    assert_size_stride(permute_567, (14336, 4096), (4096, 1))
    assert_size_stride(permute_571, (4096, 4096), (4096, 1))
    assert_size_stride(permute_577, (1024, 4096), (4096, 1))
    assert_size_stride(permute_582, (1024, 4096), (4096, 1))
    assert_size_stride(permute_587, (4096, 4096), (4096, 1))
    assert_size_stride(permute_591, (4096, 14336), (14336, 1))
    assert_size_stride(permute_595, (14336, 4096), (4096, 1))
    assert_size_stride(permute_600, (14336, 4096), (4096, 1))
    assert_size_stride(permute_604, (4096, 4096), (4096, 1))
    assert_size_stride(permute_610, (1024, 4096), (4096, 1))
    assert_size_stride(permute_615, (1024, 4096), (4096, 1))
    assert_size_stride(permute_620, (4096, 4096), (4096, 1))
    assert_size_stride(permute_624, (4096, 14336), (14336, 1))
    assert_size_stride(permute_628, (14336, 4096), (4096, 1))
    assert_size_stride(permute_633, (14336, 4096), (4096, 1))
    assert_size_stride(permute_637, (4096, 4096), (4096, 1))
    assert_size_stride(permute_643, (1024, 4096), (4096, 1))
    assert_size_stride(permute_648, (1024, 4096), (4096, 1))
    assert_size_stride(permute_653, (4096, 4096), (4096, 1))
    assert_size_stride(permute_657, (4096, 14336), (14336, 1))
    assert_size_stride(permute_661, (14336, 4096), (4096, 1))
    assert_size_stride(permute_666, (14336, 4096), (4096, 1))
    assert_size_stride(permute_670, (4096, 4096), (4096, 1))
    assert_size_stride(permute_676, (1024, 4096), (4096, 1))
    assert_size_stride(permute_681, (1024, 4096), (4096, 1))
    assert_size_stride(permute_686, (4096, 4096), (4096, 1))
    assert_size_stride(permute_690, (4096, 14336), (14336, 1))
    assert_size_stride(permute_694, (14336, 4096), (4096, 1))
    assert_size_stride(permute_699, (14336, 4096), (4096, 1))
    assert_size_stride(permute_703, (4096, 4096), (4096, 1))
    assert_size_stride(permute_709, (1024, 4096), (4096, 1))
    assert_size_stride(permute_714, (1024, 4096), (4096, 1))
    assert_size_stride(permute_719, (4096, 4096), (4096, 1))
    assert_size_stride(permute_723, (4096, 14336), (14336, 1))
    assert_size_stride(permute_727, (14336, 4096), (4096, 1))
    assert_size_stride(permute_732, (14336, 4096), (4096, 1))
    assert_size_stride(permute_736, (4096, 4096), (4096, 1))
    assert_size_stride(permute_742, (1024, 4096), (4096, 1))
    assert_size_stride(permute_747, (1024, 4096), (4096, 1))
    assert_size_stride(permute_752, (4096, 4096), (4096, 1))
    assert_size_stride(permute_756, (4096, 14336), (14336, 1))
    assert_size_stride(permute_760, (14336, 4096), (4096, 1))
    assert_size_stride(permute_765, (14336, 4096), (4096, 1))
    assert_size_stride(permute_769, (4096, 4096), (4096, 1))
    assert_size_stride(permute_775, (1024, 4096), (4096, 1))
    assert_size_stride(permute_780, (1024, 4096), (4096, 1))
    assert_size_stride(permute_785, (4096, 4096), (4096, 1))
    assert_size_stride(permute_789, (4096, 14336), (14336, 1))
    assert_size_stride(permute_793, (14336, 4096), (4096, 1))
    assert_size_stride(permute_798, (14336, 4096), (4096, 1))
    assert_size_stride(permute_802, (4096, 4096), (4096, 1))
    assert_size_stride(permute_808, (1024, 4096), (4096, 1))
    assert_size_stride(permute_813, (1024, 4096), (4096, 1))
    assert_size_stride(permute_818, (4096, 4096), (4096, 1))
    assert_size_stride(permute_822, (4096, 14336), (14336, 1))
    assert_size_stride(permute_826, (14336, 4096), (4096, 1))
    assert_size_stride(permute_831, (14336, 4096), (4096, 1))
    assert_size_stride(permute_835, (4096, 4096), (4096, 1))
    assert_size_stride(permute_841, (1024, 4096), (4096, 1))
    assert_size_stride(permute_846, (1024, 4096), (4096, 1))
    assert_size_stride(permute_851, (4096, 4096), (4096, 1))
    assert_size_stride(permute_855, (4096, 14336), (14336, 1))
    assert_size_stride(permute_859, (14336, 4096), (4096, 1))
    assert_size_stride(permute_864, (14336, 4096), (4096, 1))
    assert_size_stride(permute_868, (4096, 4096), (4096, 1))
    assert_size_stride(permute_874, (1024, 4096), (4096, 1))
    assert_size_stride(permute_879, (1024, 4096), (4096, 1))
    assert_size_stride(permute_884, (4096, 4096), (4096, 1))
    assert_size_stride(permute_888, (4096, 14336), (14336, 1))
    assert_size_stride(permute_892, (14336, 4096), (4096, 1))
    assert_size_stride(permute_897, (14336, 4096), (4096, 1))
    assert_size_stride(permute_901, (4096, 4096), (4096, 1))
    assert_size_stride(permute_907, (1024, 4096), (4096, 1))
    assert_size_stride(permute_912, (1024, 4096), (4096, 1))
    assert_size_stride(permute_917, (4096, 4096), (4096, 1))
    assert_size_stride(permute_921, (4096, 14336), (14336, 1))
    assert_size_stride(permute_925, (14336, 4096), (4096, 1))
    assert_size_stride(permute_930, (14336, 4096), (4096, 1))
    assert_size_stride(permute_934, (4096, 4096), (4096, 1))
    assert_size_stride(permute_940, (1024, 4096), (4096, 1))
    assert_size_stride(permute_945, (1024, 4096), (4096, 1))
    assert_size_stride(permute_950, (4096, 4096), (4096, 1))
    assert_size_stride(permute_954, (4096, 14336), (14336, 1))
    assert_size_stride(permute_958, (14336, 4096), (4096, 1))
    assert_size_stride(permute_963, (14336, 4096), (4096, 1))
    assert_size_stride(permute_967, (4096, 4096), (4096, 1))
    assert_size_stride(permute_973, (1024, 4096), (4096, 1))
    assert_size_stride(permute_978, (1024, 4096), (4096, 1))
    assert_size_stride(permute_983, (4096, 4096), (4096, 1))
    assert_size_stride(permute_987, (4096, 14336), (14336, 1))
    assert_size_stride(permute_991, (14336, 4096), (4096, 1))
    assert_size_stride(permute_996, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1000, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1006, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1011, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1016, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1020, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1024, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1029, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1033, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1039, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1044, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1049, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1053, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1057, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1062, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1066, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1072, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1077, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1082, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1086, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1090, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1095, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1099, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1105, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1110, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1115, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1119, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1123, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1128, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1132, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1138, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1143, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1148, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1152, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1156, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1161, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1165, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1171, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1176, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1181, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1185, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1189, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1194, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1198, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1204, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1209, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1214, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1218, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1222, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1227, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1231, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1237, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1242, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1247, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1251, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1255, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1260, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1264, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1270, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1275, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1280, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1284, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1288, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1293, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1297, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1303, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1308, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1313, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1317, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1321, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1326, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1330, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1336, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1341, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1346, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1350, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1354, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1359, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1363, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1369, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1374, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1379, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1383, (4096, 14336), (14336, 1))
    assert_size_stride(permute_1387, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1392, (14336, 4096), (4096, 1))
    assert_size_stride(permute_1396, (4096, 4096), (4096, 1))
    assert_size_stride(permute_1402, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1407, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1412, (4096, 4096), (4096, 1))
    assert_size_stride(tangents_1, (1, s0, 128256), (128256*s0, 128256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128256, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (128256, s0), (1, 128256), 0), view_649, out=buf0)
        del view_649
        buf1 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (s0, 128256), (128256, 1), 0), permute_356, out=buf1)
        del permute_356
        del tangents_1
        buf965 = empty_strided_cuda((128256, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_0.run(buf965, 525336576, stream=stream0)
        buf2 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_320, hidden_states_321, to_133], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mul_sum_1.run(buf1, add_9781, rsqrt_64, buf2, 4096, s0, stream=stream0)
        buf4 = reinterpret_tensor(buf1, (1, s0, 4096), (4096*s0, 4096, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_320], Original ATen: [aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_2.run(buf4, primals_294, add_9781, rsqrt_64, s0, 4096, stream=stream0)
        del add_9781
        del primals_294
        del rsqrt_64
        buf5 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (4096, s0), (1, 4096), 0), view_647, out=buf5)
        del view_647
        buf6 = empty_strided_cuda((s0, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (s0, 4096), (4096, 1), 0), permute_360, out=buf6)
        del permute_360
        buf7 = empty_strided_cuda((1, s0, 14336), (14336*s0, 14336, 1), torch.float16)
        buf10 = reinterpret_tensor(mm_222, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_222  # reuse
        # Topologically Sorted Source Nodes: [silu_31], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf10, buf6, mm_221, buf7, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf6
        del mm_221
        buf8 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (14336, s0), (1, 14336), 0), view_643, out=buf8)
        buf9 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (s0, 14336), (14336, 1), 0), permute_364, out=buf9)
        del permute_364
        buf11 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (14336, s0), (1, 14336), 0), view_643, out=buf11)
        del view_643
        buf12 = empty_strided_cuda((s0, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (s0, 14336), (14336, 1), 0), permute_369, out=buf12)
        del permute_369
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_316], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf15, buf9, buf12, primals_290, add_9716, rsqrt_63, s0, 4096, stream=stream0)
        del primals_290
        buf13 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_316, hidden_states_317, to_131], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf9, buf12, add_9716, rsqrt_63, buf13, 4096, s0, stream=stream0)
        del add_9716
        del buf12
        del rsqrt_63
        buf16 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_124, (s0, 4096), (4096, 1), 0), out=buf16)
        buf17 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (s0, 4096), (4096, 1), 0), permute_373, out=buf17)
        del permute_373
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf18 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf17, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_97, view_638, view_639, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_124, getitem_125, getitem_126, getitem_127, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_97
        del getitem_124
        del getitem_125
        del getitem_126
        del getitem_127
        del view_638
        del view_639
        buf19 = buf18[0]
        assert_size_stride(buf19, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf20 = buf18[1]
        assert_size_stride(buf20, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf21 = buf18[2]
        assert_size_stride(buf21, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf18
        buf29 = reinterpret_tensor(buf17, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf19, bmm, buf29, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf19
        buf22 = empty_strided_cuda((1, 8, s0, 128), (1024*s0, 128, 1024, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf20, bmm, buf22, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf26 = empty_strided_cuda((1, s0, 8, 128), (1024*s0, 1024, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf22, buf20, bmm, buf26, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf23 = reinterpret_tensor(buf22, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf21, buf23, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf30 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (4096, s0), (1, 4096), 0), view_629, out=buf30)
        buf31 = reinterpret_tensor(buf21, (s0, 4096), (4096, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0), permute_389, out=buf31)
        del permute_389
        buf27 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, s0), (1, 1024), 0), view_629, out=buf27)
        buf28 = reinterpret_tensor(buf29, (s0, 4096), (4096, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (s0, 1024), (1024, 1), 0), permute_384, out=buf28)
        del permute_384
        buf24 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1024, s0), (1, 1024), 0), view_629, out=buf24)
        del view_629
        buf25 = reinterpret_tensor(buf20, (s0, 4096), (4096, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (s0, 1024), (1024, 1), 0), permute_379, out=buf25)
        del permute_379
        buf34 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_310], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf34, buf25, buf28, buf31, primals_285, add_9478, rsqrt_62, s0, 4096, stream=stream0)
        del primals_285
        buf32 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_310, hidden_states_311, to_129], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf25, buf28, buf31, add_9478, rsqrt_62, buf32, 4096, s0, stream=stream0)
        del add_9478
        del buf25
        del rsqrt_62
        buf35 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (4096, s0), (1, 4096), 0), view_627, out=buf35)
        del view_627
        buf36 = reinterpret_tensor(buf10, (s0, 14336), (14336, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (s0, 4096), (4096, 1), 0), permute_393, out=buf36)
        del permute_393
        buf37 = buf7; del buf7  # reuse
        buf40 = reinterpret_tensor(mm_215, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_215  # reuse
        # Topologically Sorted Source Nodes: [silu_30], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf40, buf36, mm_214, buf37, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf36
        del mm_214
        buf38 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (14336, s0), (1, 14336), 0), view_623, out=buf38)
        buf39 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (s0, 14336), (14336, 1), 0), permute_397, out=buf39)
        del permute_397
        buf41 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (14336, s0), (1, 14336), 0), view_623, out=buf41)
        del view_623
        buf42 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (s0, 14336), (14336, 1), 0), permute_402, out=buf42)
        del permute_402
        buf45 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_306], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf45, buf39, buf42, primals_281, add_9413, rsqrt_61, s0, 4096, stream=stream0)
        del primals_281
        buf43 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_306, hidden_states_307, to_127], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf39, buf42, add_9413, rsqrt_61, buf43, 4096, s0, stream=stream0)
        del add_9413
        del buf39
        del rsqrt_61
        buf46 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_120, (s0, 4096), (4096, 1), 0), out=buf46)
        buf47 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (s0, 4096), (4096, 1), 0), permute_406, out=buf47)
        del permute_406
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf48 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf47, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_94, view_618, view_619, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_120, getitem_121, getitem_122, getitem_123, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_94
        del getitem_120
        del getitem_121
        del getitem_122
        del getitem_123
        del view_618
        del view_619
        buf49 = buf48[0]
        assert_size_stride(buf49, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf50 = buf48[1]
        assert_size_stride(buf50, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf51 = buf48[2]
        assert_size_stride(buf51, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf48
        buf59 = reinterpret_tensor(buf47, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf49, bmm, buf59, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf49
        buf52 = reinterpret_tensor(buf23, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf50, bmm, buf52, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf56 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf52, buf50, bmm, buf56, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf53 = reinterpret_tensor(buf52, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf51, buf53, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf60 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (4096, s0), (1, 4096), 0), view_609, out=buf60)
        buf61 = reinterpret_tensor(buf51, (s0, 4096), (4096, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (s0, 4096), (4096, 1), 0), permute_422, out=buf61)
        del permute_422
        buf57 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (1024, s0), (1, 1024), 0), view_609, out=buf57)
        buf58 = reinterpret_tensor(buf59, (s0, 4096), (4096, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (s0, 1024), (1024, 1), 0), permute_417, out=buf58)
        del permute_417
        buf54 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1024, s0), (1, 1024), 0), view_609, out=buf54)
        del view_609
        buf55 = reinterpret_tensor(buf50, (s0, 4096), (4096, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (s0, 1024), (1024, 1), 0), permute_412, out=buf55)
        del permute_412
        buf64 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_300], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf64, buf55, buf58, buf61, primals_276, add_9175, rsqrt_60, s0, 4096, stream=stream0)
        del primals_276
        buf62 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_300, hidden_states_301, to_125], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf55, buf58, buf61, add_9175, rsqrt_60, buf62, 4096, s0, stream=stream0)
        del add_9175
        del buf55
        del rsqrt_60
        buf65 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (4096, s0), (1, 4096), 0), view_607, out=buf65)
        del view_607
        buf66 = reinterpret_tensor(buf40, (s0, 14336), (14336, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (s0, 4096), (4096, 1), 0), permute_426, out=buf66)
        del permute_426
        buf67 = buf37; del buf37  # reuse
        buf70 = reinterpret_tensor(mm_208, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_208  # reuse
        # Topologically Sorted Source Nodes: [silu_29], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf70, buf66, mm_207, buf67, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf66
        del mm_207
        buf68 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (14336, s0), (1, 14336), 0), view_603, out=buf68)
        buf69 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (s0, 14336), (14336, 1), 0), permute_430, out=buf69)
        del permute_430
        buf71 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (14336, s0), (1, 14336), 0), view_603, out=buf71)
        del view_603
        buf72 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (s0, 14336), (14336, 1), 0), permute_435, out=buf72)
        del permute_435
        buf75 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_296], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf75, buf69, buf72, primals_272, add_9110, rsqrt_59, s0, 4096, stream=stream0)
        del primals_272
        buf73 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_296, hidden_states_297, to_123], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf69, buf72, add_9110, rsqrt_59, buf73, 4096, s0, stream=stream0)
        del add_9110
        del buf69
        del rsqrt_59
        buf76 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_116, (s0, 4096), (4096, 1), 0), out=buf76)
        buf77 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (s0, 4096), (4096, 1), 0), permute_439, out=buf77)
        del permute_439
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf78 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf77, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_91, view_598, view_599, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_116, getitem_117, getitem_118, getitem_119, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_91
        del getitem_116
        del getitem_117
        del getitem_118
        del getitem_119
        del view_598
        del view_599
        buf79 = buf78[0]
        assert_size_stride(buf79, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf80 = buf78[1]
        assert_size_stride(buf80, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf81 = buf78[2]
        assert_size_stride(buf81, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf78
        buf89 = reinterpret_tensor(buf77, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf79, bmm, buf89, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf79
        buf82 = reinterpret_tensor(buf53, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf80, bmm, buf82, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf86 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf82, buf80, bmm, buf86, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf83 = reinterpret_tensor(buf82, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf81, buf83, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf90 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (4096, s0), (1, 4096), 0), view_589, out=buf90)
        buf91 = reinterpret_tensor(buf81, (s0, 4096), (4096, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (s0, 4096), (4096, 1), 0), permute_455, out=buf91)
        del permute_455
        buf87 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1024, s0), (1, 1024), 0), view_589, out=buf87)
        buf88 = reinterpret_tensor(buf89, (s0, 4096), (4096, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (s0, 1024), (1024, 1), 0), permute_450, out=buf88)
        del permute_450
        buf84 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (1024, s0), (1, 1024), 0), view_589, out=buf84)
        del view_589
        buf85 = reinterpret_tensor(buf80, (s0, 4096), (4096, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (s0, 1024), (1024, 1), 0), permute_445, out=buf85)
        del permute_445
        buf94 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_290], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf94, buf85, buf88, buf91, primals_267, add_8872, rsqrt_58, s0, 4096, stream=stream0)
        del primals_267
        buf92 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_290, hidden_states_291, to_121], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf85, buf88, buf91, add_8872, rsqrt_58, buf92, 4096, s0, stream=stream0)
        del add_8872
        del buf85
        del rsqrt_58
        buf95 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (4096, s0), (1, 4096), 0), view_587, out=buf95)
        del view_587
        buf96 = reinterpret_tensor(buf70, (s0, 14336), (14336, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (s0, 4096), (4096, 1), 0), permute_459, out=buf96)
        del permute_459
        buf97 = buf67; del buf67  # reuse
        buf100 = reinterpret_tensor(mm_201, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_201  # reuse
        # Topologically Sorted Source Nodes: [silu_28], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf100, buf96, mm_200, buf97, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf96
        del mm_200
        buf98 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (14336, s0), (1, 14336), 0), view_583, out=buf98)
        buf99 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (s0, 14336), (14336, 1), 0), permute_463, out=buf99)
        del permute_463
        buf101 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (14336, s0), (1, 14336), 0), view_583, out=buf101)
        del view_583
        buf102 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (s0, 14336), (14336, 1), 0), permute_468, out=buf102)
        del permute_468
        buf105 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_286], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf105, buf99, buf102, primals_263, add_8807, rsqrt_57, s0, 4096, stream=stream0)
        del primals_263
        buf103 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_286, hidden_states_287, to_119], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf99, buf102, add_8807, rsqrt_57, buf103, 4096, s0, stream=stream0)
        del add_8807
        del buf102
        del rsqrt_57
        buf106 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_112, (s0, 4096), (4096, 1), 0), out=buf106)
        buf107 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (s0, 4096), (4096, 1), 0), permute_472, out=buf107)
        del permute_472
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf108 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf107, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_88, view_578, view_579, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_112, getitem_113, getitem_114, getitem_115, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_88
        del getitem_112
        del getitem_113
        del getitem_114
        del getitem_115
        del view_578
        del view_579
        buf109 = buf108[0]
        assert_size_stride(buf109, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf110 = buf108[1]
        assert_size_stride(buf110, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf111 = buf108[2]
        assert_size_stride(buf111, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf108
        buf119 = reinterpret_tensor(buf107, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf109, bmm, buf119, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf109
        buf112 = reinterpret_tensor(buf83, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf110, bmm, buf112, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf116 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf112, buf110, bmm, buf116, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf113 = reinterpret_tensor(buf112, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf111, buf113, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf120 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (4096, s0), (1, 4096), 0), view_569, out=buf120)
        buf121 = reinterpret_tensor(buf111, (s0, 4096), (4096, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (s0, 4096), (4096, 1), 0), permute_488, out=buf121)
        del permute_488
        buf117 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, s0), (1, 1024), 0), view_569, out=buf117)
        buf118 = reinterpret_tensor(buf119, (s0, 4096), (4096, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (s0, 1024), (1024, 1), 0), permute_483, out=buf118)
        del permute_483
        buf114 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1024, s0), (1, 1024), 0), view_569, out=buf114)
        del view_569
        buf115 = reinterpret_tensor(buf110, (s0, 4096), (4096, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (s0, 1024), (1024, 1), 0), permute_478, out=buf115)
        del permute_478
        buf124 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_280], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf124, buf115, buf118, buf121, primals_258, add_8569, rsqrt_56, s0, 4096, stream=stream0)
        del primals_258
        buf122 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_280, hidden_states_281, to_117], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf115, buf118, buf121, add_8569, rsqrt_56, buf122, 4096, s0, stream=stream0)
        del add_8569
        del buf115
        del rsqrt_56
        buf125 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (4096, s0), (1, 4096), 0), view_567, out=buf125)
        del view_567
        buf126 = reinterpret_tensor(buf100, (s0, 14336), (14336, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (s0, 4096), (4096, 1), 0), permute_492, out=buf126)
        del permute_492
        buf127 = buf97; del buf97  # reuse
        buf130 = reinterpret_tensor(mm_194, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_194  # reuse
        # Topologically Sorted Source Nodes: [silu_27], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf130, buf126, mm_193, buf127, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf126
        del mm_193
        buf128 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (14336, s0), (1, 14336), 0), view_563, out=buf128)
        buf129 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (s0, 14336), (14336, 1), 0), permute_496, out=buf129)
        del permute_496
        buf131 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (14336, s0), (1, 14336), 0), view_563, out=buf131)
        del view_563
        buf132 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (s0, 14336), (14336, 1), 0), permute_501, out=buf132)
        del permute_501
        buf135 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_276], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf135, buf129, buf132, primals_254, add_8504, rsqrt_55, s0, 4096, stream=stream0)
        del primals_254
        buf133 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_276, hidden_states_277, to_115], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf129, buf132, add_8504, rsqrt_55, buf133, 4096, s0, stream=stream0)
        del add_8504
        del buf129
        del rsqrt_55
        buf136 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_108, (s0, 4096), (4096, 1), 0), out=buf136)
        buf137 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (s0, 4096), (4096, 1), 0), permute_505, out=buf137)
        del permute_505
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf138 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf137, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_85, view_558, view_559, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_108, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_85
        del getitem_108
        del getitem_109
        del getitem_110
        del getitem_111
        del view_558
        del view_559
        buf139 = buf138[0]
        assert_size_stride(buf139, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf140 = buf138[1]
        assert_size_stride(buf140, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf141 = buf138[2]
        assert_size_stride(buf141, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf138
        buf149 = reinterpret_tensor(buf137, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf139, bmm, buf149, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf139
        buf142 = reinterpret_tensor(buf113, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf140, bmm, buf142, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf146 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf142, buf140, bmm, buf146, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf143 = reinterpret_tensor(buf142, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf141, buf143, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf150 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (4096, s0), (1, 4096), 0), view_549, out=buf150)
        buf151 = reinterpret_tensor(buf141, (s0, 4096), (4096, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0), permute_521, out=buf151)
        del permute_521
        buf147 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (1024, s0), (1, 1024), 0), view_549, out=buf147)
        buf148 = reinterpret_tensor(buf149, (s0, 4096), (4096, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (s0, 1024), (1024, 1), 0), permute_516, out=buf148)
        del permute_516
        buf144 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (1024, s0), (1, 1024), 0), view_549, out=buf144)
        del view_549
        buf145 = reinterpret_tensor(buf140, (s0, 4096), (4096, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (s0, 1024), (1024, 1), 0), permute_511, out=buf145)
        del permute_511
        buf154 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_270], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf154, buf145, buf148, buf151, primals_249, add_8266, rsqrt_54, s0, 4096, stream=stream0)
        del primals_249
        buf152 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_270, hidden_states_271, to_113], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf145, buf148, buf151, add_8266, rsqrt_54, buf152, 4096, s0, stream=stream0)
        del add_8266
        del buf145
        del rsqrt_54
        buf155 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (4096, s0), (1, 4096), 0), view_547, out=buf155)
        del view_547
        buf156 = reinterpret_tensor(buf130, (s0, 14336), (14336, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (s0, 4096), (4096, 1), 0), permute_525, out=buf156)
        del permute_525
        buf157 = buf127; del buf127  # reuse
        buf160 = reinterpret_tensor(mm_187, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_187  # reuse
        # Topologically Sorted Source Nodes: [silu_26], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf160, buf156, mm_186, buf157, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf156
        del mm_186
        buf158 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (14336, s0), (1, 14336), 0), view_543, out=buf158)
        buf159 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (s0, 14336), (14336, 1), 0), permute_529, out=buf159)
        del permute_529
        buf161 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (14336, s0), (1, 14336), 0), view_543, out=buf161)
        del view_543
        buf162 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (s0, 14336), (14336, 1), 0), permute_534, out=buf162)
        del permute_534
        buf165 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_266], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf165, buf159, buf162, primals_245, add_8201, rsqrt_53, s0, 4096, stream=stream0)
        del primals_245
        buf163 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_266, hidden_states_267, to_111], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf159, buf162, add_8201, rsqrt_53, buf163, 4096, s0, stream=stream0)
        del add_8201
        del buf159
        del rsqrt_53
        buf166 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_104, (s0, 4096), (4096, 1), 0), out=buf166)
        buf167 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (s0, 4096), (4096, 1), 0), permute_538, out=buf167)
        del permute_538
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf167, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_82, view_538, view_539, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_104, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_82
        del getitem_104
        del getitem_105
        del getitem_106
        del getitem_107
        del view_538
        del view_539
        buf169 = buf168[0]
        assert_size_stride(buf169, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf170 = buf168[1]
        assert_size_stride(buf170, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf171 = buf168[2]
        assert_size_stride(buf171, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf168
        buf179 = reinterpret_tensor(buf167, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf169, bmm, buf179, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf169
        buf172 = reinterpret_tensor(buf143, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf170, bmm, buf172, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf176 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf172, buf170, bmm, buf176, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf173 = reinterpret_tensor(buf172, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf171, buf173, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf180 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (4096, s0), (1, 4096), 0), view_529, out=buf180)
        buf181 = reinterpret_tensor(buf171, (s0, 4096), (4096, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (s0, 4096), (4096, 1), 0), permute_554, out=buf181)
        del permute_554
        buf177 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (1024, s0), (1, 1024), 0), view_529, out=buf177)
        buf178 = reinterpret_tensor(buf179, (s0, 4096), (4096, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (s0, 1024), (1024, 1), 0), permute_549, out=buf178)
        del permute_549
        buf174 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (1024, s0), (1, 1024), 0), view_529, out=buf174)
        del view_529
        buf175 = reinterpret_tensor(buf170, (s0, 4096), (4096, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (s0, 1024), (1024, 1), 0), permute_544, out=buf175)
        del permute_544
        buf184 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_260], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf184, buf175, buf178, buf181, primals_240, add_7963, rsqrt_52, s0, 4096, stream=stream0)
        del primals_240
        buf182 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261, to_109], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf175, buf178, buf181, add_7963, rsqrt_52, buf182, 4096, s0, stream=stream0)
        del add_7963
        del buf175
        del rsqrt_52
        buf185 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (4096, s0), (1, 4096), 0), view_527, out=buf185)
        del view_527
        buf186 = reinterpret_tensor(buf160, (s0, 14336), (14336, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (s0, 4096), (4096, 1), 0), permute_558, out=buf186)
        del permute_558
        buf187 = buf157; del buf157  # reuse
        buf190 = reinterpret_tensor(mm_180, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_180  # reuse
        # Topologically Sorted Source Nodes: [silu_25], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf190, buf186, mm_179, buf187, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf186
        del mm_179
        buf188 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (14336, s0), (1, 14336), 0), view_523, out=buf188)
        buf189 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (s0, 14336), (14336, 1), 0), permute_562, out=buf189)
        del permute_562
        buf191 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (14336, s0), (1, 14336), 0), view_523, out=buf191)
        del view_523
        buf192 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (s0, 14336), (14336, 1), 0), permute_567, out=buf192)
        del permute_567
        buf195 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_256], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf195, buf189, buf192, primals_236, add_7898, rsqrt_51, s0, 4096, stream=stream0)
        del primals_236
        buf193 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_256, hidden_states_257, to_107], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf189, buf192, add_7898, rsqrt_51, buf193, 4096, s0, stream=stream0)
        del add_7898
        del buf189
        del rsqrt_51
        buf196 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_100, (s0, 4096), (4096, 1), 0), out=buf196)
        buf197 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (s0, 4096), (4096, 1), 0), permute_571, out=buf197)
        del permute_571
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf198 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf197, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_79, view_518, view_519, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_100, getitem_101, getitem_102, getitem_103, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_79
        del getitem_100
        del getitem_101
        del getitem_102
        del getitem_103
        del view_518
        del view_519
        buf199 = buf198[0]
        assert_size_stride(buf199, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf200 = buf198[1]
        assert_size_stride(buf200, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf201 = buf198[2]
        assert_size_stride(buf201, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf198
        buf209 = reinterpret_tensor(buf197, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf199, bmm, buf209, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf199
        buf202 = reinterpret_tensor(buf173, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf200, bmm, buf202, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf206 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf202, buf200, bmm, buf206, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf203 = reinterpret_tensor(buf202, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf201, buf203, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf210 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (4096, s0), (1, 4096), 0), view_509, out=buf210)
        buf211 = reinterpret_tensor(buf201, (s0, 4096), (4096, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (s0, 4096), (4096, 1), 0), permute_587, out=buf211)
        del permute_587
        buf207 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (1024, s0), (1, 1024), 0), view_509, out=buf207)
        buf208 = reinterpret_tensor(buf209, (s0, 4096), (4096, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (s0, 1024), (1024, 1), 0), permute_582, out=buf208)
        del permute_582
        buf204 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (1024, s0), (1, 1024), 0), view_509, out=buf204)
        del view_509
        buf205 = reinterpret_tensor(buf200, (s0, 4096), (4096, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (s0, 1024), (1024, 1), 0), permute_577, out=buf205)
        del permute_577
        buf214 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_250], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf214, buf205, buf208, buf211, primals_231, add_7660, rsqrt_50, s0, 4096, stream=stream0)
        del primals_231
        buf212 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251, to_105], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf205, buf208, buf211, add_7660, rsqrt_50, buf212, 4096, s0, stream=stream0)
        del add_7660
        del buf205
        del rsqrt_50
        buf215 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (4096, s0), (1, 4096), 0), view_507, out=buf215)
        del view_507
        buf216 = reinterpret_tensor(buf190, (s0, 14336), (14336, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (s0, 4096), (4096, 1), 0), permute_591, out=buf216)
        del permute_591
        buf217 = buf187; del buf187  # reuse
        buf220 = reinterpret_tensor(mm_173, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_173  # reuse
        # Topologically Sorted Source Nodes: [silu_24], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf220, buf216, mm_172, buf217, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf216
        del mm_172
        buf218 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (14336, s0), (1, 14336), 0), view_503, out=buf218)
        buf219 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (s0, 14336), (14336, 1), 0), permute_595, out=buf219)
        del permute_595
        buf221 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (14336, s0), (1, 14336), 0), view_503, out=buf221)
        del view_503
        buf222 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (s0, 14336), (14336, 1), 0), permute_600, out=buf222)
        del permute_600
        buf225 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_246], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf225, buf219, buf222, primals_227, add_7595, rsqrt_49, s0, 4096, stream=stream0)
        del primals_227
        buf223 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_246, hidden_states_247, to_103], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf219, buf222, add_7595, rsqrt_49, buf223, 4096, s0, stream=stream0)
        del add_7595
        del buf219
        del rsqrt_49
        buf226 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_96, (s0, 4096), (4096, 1), 0), out=buf226)
        buf227 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (s0, 4096), (4096, 1), 0), permute_604, out=buf227)
        del permute_604
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf228 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf227, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_76, view_498, view_499, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_96, getitem_97, getitem_98, getitem_99, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_76
        del getitem_96
        del getitem_97
        del getitem_98
        del getitem_99
        del view_498
        del view_499
        buf229 = buf228[0]
        assert_size_stride(buf229, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf230 = buf228[1]
        assert_size_stride(buf230, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf231 = buf228[2]
        assert_size_stride(buf231, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf228
        buf239 = reinterpret_tensor(buf227, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf229, bmm, buf239, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf229
        buf232 = reinterpret_tensor(buf203, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf230, bmm, buf232, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf236 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf232, buf230, bmm, buf236, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf233 = reinterpret_tensor(buf232, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf231, buf233, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf240 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (4096, s0), (1, 4096), 0), view_489, out=buf240)
        buf241 = reinterpret_tensor(buf231, (s0, 4096), (4096, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (s0, 4096), (4096, 1), 0), permute_620, out=buf241)
        del permute_620
        buf237 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (1024, s0), (1, 1024), 0), view_489, out=buf237)
        buf238 = reinterpret_tensor(buf239, (s0, 4096), (4096, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (s0, 1024), (1024, 1), 0), permute_615, out=buf238)
        del permute_615
        buf234 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1024, s0), (1, 1024), 0), view_489, out=buf234)
        del view_489
        buf235 = reinterpret_tensor(buf230, (s0, 4096), (4096, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (s0, 1024), (1024, 1), 0), permute_610, out=buf235)
        del permute_610
        buf244 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_240], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf244, buf235, buf238, buf241, primals_222, add_7357, rsqrt_48, s0, 4096, stream=stream0)
        del primals_222
        buf242 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_240, hidden_states_241, to_101], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf235, buf238, buf241, add_7357, rsqrt_48, buf242, 4096, s0, stream=stream0)
        del add_7357
        del buf235
        del rsqrt_48
        buf245 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (4096, s0), (1, 4096), 0), view_487, out=buf245)
        del view_487
        buf246 = reinterpret_tensor(buf220, (s0, 14336), (14336, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (s0, 4096), (4096, 1), 0), permute_624, out=buf246)
        del permute_624
        buf247 = buf217; del buf217  # reuse
        buf250 = reinterpret_tensor(mm_166, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_166  # reuse
        # Topologically Sorted Source Nodes: [silu_23], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf250, buf246, mm_165, buf247, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf246
        del mm_165
        buf248 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (14336, s0), (1, 14336), 0), view_483, out=buf248)
        buf249 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (s0, 14336), (14336, 1), 0), permute_628, out=buf249)
        del permute_628
        buf251 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (14336, s0), (1, 14336), 0), view_483, out=buf251)
        del view_483
        buf252 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (s0, 14336), (14336, 1), 0), permute_633, out=buf252)
        del permute_633
        buf255 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_236], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf255, buf249, buf252, primals_218, add_7292, rsqrt_47, s0, 4096, stream=stream0)
        del primals_218
        buf253 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_236, hidden_states_237, to_99], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf249, buf252, add_7292, rsqrt_47, buf253, 4096, s0, stream=stream0)
        del add_7292
        del buf249
        del rsqrt_47
        buf256 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_92, (s0, 4096), (4096, 1), 0), out=buf256)
        buf257 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (s0, 4096), (4096, 1), 0), permute_637, out=buf257)
        del permute_637
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf258 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf257, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_73, view_478, view_479, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_92, getitem_93, getitem_94, getitem_95, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_73
        del getitem_92
        del getitem_93
        del getitem_94
        del getitem_95
        del view_478
        del view_479
        buf259 = buf258[0]
        assert_size_stride(buf259, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf260 = buf258[1]
        assert_size_stride(buf260, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf261 = buf258[2]
        assert_size_stride(buf261, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf258
        buf269 = reinterpret_tensor(buf257, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf259, bmm, buf269, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf259
        buf262 = reinterpret_tensor(buf233, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf260, bmm, buf262, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf266 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf262, buf260, bmm, buf266, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf263 = reinterpret_tensor(buf262, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf261, buf263, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf270 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (4096, s0), (1, 4096), 0), view_469, out=buf270)
        buf271 = reinterpret_tensor(buf261, (s0, 4096), (4096, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0), permute_653, out=buf271)
        del permute_653
        buf267 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (1024, s0), (1, 1024), 0), view_469, out=buf267)
        buf268 = reinterpret_tensor(buf269, (s0, 4096), (4096, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (s0, 1024), (1024, 1), 0), permute_648, out=buf268)
        del permute_648
        buf264 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (1024, s0), (1, 1024), 0), view_469, out=buf264)
        del view_469
        buf265 = reinterpret_tensor(buf260, (s0, 4096), (4096, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (s0, 1024), (1024, 1), 0), permute_643, out=buf265)
        del permute_643
        buf274 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_230], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf274, buf265, buf268, buf271, primals_213, add_7054, rsqrt_46, s0, 4096, stream=stream0)
        del primals_213
        buf272 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_97], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf265, buf268, buf271, add_7054, rsqrt_46, buf272, 4096, s0, stream=stream0)
        del add_7054
        del buf265
        del rsqrt_46
        buf275 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (4096, s0), (1, 4096), 0), view_467, out=buf275)
        del view_467
        buf276 = reinterpret_tensor(buf250, (s0, 14336), (14336, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (s0, 4096), (4096, 1), 0), permute_657, out=buf276)
        del permute_657
        buf277 = buf247; del buf247  # reuse
        buf280 = reinterpret_tensor(mm_159, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_159  # reuse
        # Topologically Sorted Source Nodes: [silu_22], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf280, buf276, mm_158, buf277, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf276
        del mm_158
        buf278 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (14336, s0), (1, 14336), 0), view_463, out=buf278)
        buf279 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (s0, 14336), (14336, 1), 0), permute_661, out=buf279)
        del permute_661
        buf281 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (14336, s0), (1, 14336), 0), view_463, out=buf281)
        del view_463
        buf282 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (s0, 14336), (14336, 1), 0), permute_666, out=buf282)
        del permute_666
        buf285 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_226], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf285, buf279, buf282, primals_209, add_6989, rsqrt_45, s0, 4096, stream=stream0)
        del primals_209
        buf283 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227, to_95], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf279, buf282, add_6989, rsqrt_45, buf283, 4096, s0, stream=stream0)
        del add_6989
        del buf279
        del rsqrt_45
        buf286 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_88, (s0, 4096), (4096, 1), 0), out=buf286)
        buf287 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (s0, 4096), (4096, 1), 0), permute_670, out=buf287)
        del permute_670
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf288 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf287, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_70, view_458, view_459, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_88, getitem_89, getitem_90, getitem_91, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_70
        del getitem_88
        del getitem_89
        del getitem_90
        del getitem_91
        del view_458
        del view_459
        buf289 = buf288[0]
        assert_size_stride(buf289, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf290 = buf288[1]
        assert_size_stride(buf290, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf291 = buf288[2]
        assert_size_stride(buf291, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf288
        buf299 = reinterpret_tensor(buf287, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf289, bmm, buf299, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf289
        buf292 = reinterpret_tensor(buf263, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf290, bmm, buf292, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf296 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf292, buf290, bmm, buf296, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf293 = reinterpret_tensor(buf292, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf291, buf293, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf300 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (4096, s0), (1, 4096), 0), view_449, out=buf300)
        buf301 = reinterpret_tensor(buf291, (s0, 4096), (4096, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (s0, 4096), (4096, 1), 0), permute_686, out=buf301)
        del permute_686
        buf297 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (1024, s0), (1, 1024), 0), view_449, out=buf297)
        buf298 = reinterpret_tensor(buf299, (s0, 4096), (4096, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (s0, 1024), (1024, 1), 0), permute_681, out=buf298)
        del permute_681
        buf294 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (1024, s0), (1, 1024), 0), view_449, out=buf294)
        del view_449
        buf295 = reinterpret_tensor(buf290, (s0, 4096), (4096, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (s0, 1024), (1024, 1), 0), permute_676, out=buf295)
        del permute_676
        buf304 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_220], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf304, buf295, buf298, buf301, primals_204, add_6751, rsqrt_44, s0, 4096, stream=stream0)
        del primals_204
        buf302 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_93], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf295, buf298, buf301, add_6751, rsqrt_44, buf302, 4096, s0, stream=stream0)
        del add_6751
        del buf295
        del rsqrt_44
        buf305 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (4096, s0), (1, 4096), 0), view_447, out=buf305)
        del view_447
        buf306 = reinterpret_tensor(buf280, (s0, 14336), (14336, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (s0, 4096), (4096, 1), 0), permute_690, out=buf306)
        del permute_690
        buf307 = buf277; del buf277  # reuse
        buf310 = reinterpret_tensor(mm_152, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_152  # reuse
        # Topologically Sorted Source Nodes: [silu_21], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf310, buf306, mm_151, buf307, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf306
        del mm_151
        buf308 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (14336, s0), (1, 14336), 0), view_443, out=buf308)
        buf309 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (s0, 14336), (14336, 1), 0), permute_694, out=buf309)
        del permute_694
        buf311 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (14336, s0), (1, 14336), 0), view_443, out=buf311)
        del view_443
        buf312 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (s0, 14336), (14336, 1), 0), permute_699, out=buf312)
        del permute_699
        buf315 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_216], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf315, buf309, buf312, primals_200, add_6686, rsqrt_43, s0, 4096, stream=stream0)
        del primals_200
        buf313 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_216, hidden_states_217, to_91], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf309, buf312, add_6686, rsqrt_43, buf313, 4096, s0, stream=stream0)
        del add_6686
        del buf309
        del rsqrt_43
        buf316 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_84, (s0, 4096), (4096, 1), 0), out=buf316)
        buf317 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (s0, 4096), (4096, 1), 0), permute_703, out=buf317)
        del permute_703
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf318 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf317, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_67, view_438, view_439, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_84, getitem_85, getitem_86, getitem_87, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_67
        del getitem_84
        del getitem_85
        del getitem_86
        del getitem_87
        del view_438
        del view_439
        buf319 = buf318[0]
        assert_size_stride(buf319, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf320 = buf318[1]
        assert_size_stride(buf320, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf321 = buf318[2]
        assert_size_stride(buf321, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf318
        buf329 = reinterpret_tensor(buf317, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf319, bmm, buf329, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf319
        buf322 = reinterpret_tensor(buf293, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf320, bmm, buf322, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf326 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf322, buf320, bmm, buf326, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf323 = reinterpret_tensor(buf322, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf321, buf323, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf330 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (4096, s0), (1, 4096), 0), view_429, out=buf330)
        buf331 = reinterpret_tensor(buf321, (s0, 4096), (4096, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (s0, 4096), (4096, 1), 0), permute_719, out=buf331)
        del permute_719
        buf327 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (1024, s0), (1, 1024), 0), view_429, out=buf327)
        buf328 = reinterpret_tensor(buf329, (s0, 4096), (4096, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (s0, 1024), (1024, 1), 0), permute_714, out=buf328)
        del permute_714
        buf324 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (1024, s0), (1, 1024), 0), view_429, out=buf324)
        del view_429
        buf325 = reinterpret_tensor(buf320, (s0, 4096), (4096, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (s0, 1024), (1024, 1), 0), permute_709, out=buf325)
        del permute_709
        buf334 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_210], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf334, buf325, buf328, buf331, primals_195, add_6448, rsqrt_42, s0, 4096, stream=stream0)
        del primals_195
        buf332 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_89], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf325, buf328, buf331, add_6448, rsqrt_42, buf332, 4096, s0, stream=stream0)
        del add_6448
        del buf325
        del rsqrt_42
        buf335 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (4096, s0), (1, 4096), 0), view_427, out=buf335)
        del view_427
        buf336 = reinterpret_tensor(buf310, (s0, 14336), (14336, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (s0, 4096), (4096, 1), 0), permute_723, out=buf336)
        del permute_723
        buf337 = buf307; del buf307  # reuse
        buf340 = reinterpret_tensor(mm_145, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_145  # reuse
        # Topologically Sorted Source Nodes: [silu_20], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf340, buf336, mm_144, buf337, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf336
        del mm_144
        buf338 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (14336, s0), (1, 14336), 0), view_423, out=buf338)
        buf339 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (s0, 14336), (14336, 1), 0), permute_727, out=buf339)
        del permute_727
        buf341 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (14336, s0), (1, 14336), 0), view_423, out=buf341)
        del view_423
        buf342 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (s0, 14336), (14336, 1), 0), permute_732, out=buf342)
        del permute_732
        buf345 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_206], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf345, buf339, buf342, primals_191, add_6383, rsqrt_41, s0, 4096, stream=stream0)
        del primals_191
        buf343 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_207, to_87], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf339, buf342, add_6383, rsqrt_41, buf343, 4096, s0, stream=stream0)
        del add_6383
        del buf339
        del rsqrt_41
        buf346 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_80, (s0, 4096), (4096, 1), 0), out=buf346)
        buf347 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (s0, 4096), (4096, 1), 0), permute_736, out=buf347)
        del permute_736
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf347, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_64, view_418, view_419, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_80, getitem_81, getitem_82, getitem_83, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_64
        del getitem_80
        del getitem_81
        del getitem_82
        del getitem_83
        del view_418
        del view_419
        buf349 = buf348[0]
        assert_size_stride(buf349, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf350 = buf348[1]
        assert_size_stride(buf350, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf351 = buf348[2]
        assert_size_stride(buf351, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf348
        buf359 = reinterpret_tensor(buf347, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf349, bmm, buf359, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf349
        buf352 = reinterpret_tensor(buf323, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf350, bmm, buf352, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf356 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf352, buf350, bmm, buf356, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf353 = reinterpret_tensor(buf352, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf351, buf353, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf360 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (4096, s0), (1, 4096), 0), view_409, out=buf360)
        buf361 = reinterpret_tensor(buf351, (s0, 4096), (4096, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (s0, 4096), (4096, 1), 0), permute_752, out=buf361)
        del permute_752
        buf357 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (1024, s0), (1, 1024), 0), view_409, out=buf357)
        buf358 = reinterpret_tensor(buf359, (s0, 4096), (4096, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (s0, 1024), (1024, 1), 0), permute_747, out=buf358)
        del permute_747
        buf354 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1024, s0), (1, 1024), 0), view_409, out=buf354)
        del view_409
        buf355 = reinterpret_tensor(buf350, (s0, 4096), (4096, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (s0, 1024), (1024, 1), 0), permute_742, out=buf355)
        del permute_742
        buf364 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_200], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf364, buf355, buf358, buf361, primals_186, add_6145, rsqrt_40, s0, 4096, stream=stream0)
        del primals_186
        buf362 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_85], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf355, buf358, buf361, add_6145, rsqrt_40, buf362, 4096, s0, stream=stream0)
        del add_6145
        del buf355
        del rsqrt_40
        buf365 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (4096, s0), (1, 4096), 0), view_407, out=buf365)
        del view_407
        buf366 = reinterpret_tensor(buf340, (s0, 14336), (14336, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (s0, 4096), (4096, 1), 0), permute_756, out=buf366)
        del permute_756
        buf367 = buf337; del buf337  # reuse
        buf370 = reinterpret_tensor(mm_138, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_138  # reuse
        # Topologically Sorted Source Nodes: [silu_19], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf370, buf366, mm_137, buf367, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf366
        del mm_137
        buf368 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (14336, s0), (1, 14336), 0), view_403, out=buf368)
        buf369 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (s0, 14336), (14336, 1), 0), permute_760, out=buf369)
        del permute_760
        buf371 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (14336, s0), (1, 14336), 0), view_403, out=buf371)
        del view_403
        buf372 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (s0, 14336), (14336, 1), 0), permute_765, out=buf372)
        del permute_765
        buf375 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_196], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf375, buf369, buf372, primals_182, add_6080, rsqrt_39, s0, 4096, stream=stream0)
        del primals_182
        buf373 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197, to_83], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf369, buf372, add_6080, rsqrt_39, buf373, 4096, s0, stream=stream0)
        del add_6080
        del buf369
        del rsqrt_39
        buf376 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_76, (s0, 4096), (4096, 1), 0), out=buf376)
        buf377 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (s0, 4096), (4096, 1), 0), permute_769, out=buf377)
        del permute_769
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf378 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf377, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_61, view_398, view_399, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_76, getitem_77, getitem_78, getitem_79, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_61
        del getitem_76
        del getitem_77
        del getitem_78
        del getitem_79
        del view_398
        del view_399
        buf379 = buf378[0]
        assert_size_stride(buf379, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf380 = buf378[1]
        assert_size_stride(buf380, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf381 = buf378[2]
        assert_size_stride(buf381, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf378
        buf389 = reinterpret_tensor(buf377, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf379, bmm, buf389, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf379
        buf382 = reinterpret_tensor(buf353, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf380, bmm, buf382, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf386 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf382, buf380, bmm, buf386, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf383 = reinterpret_tensor(buf382, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf381, buf383, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf390 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (4096, s0), (1, 4096), 0), view_389, out=buf390)
        buf391 = reinterpret_tensor(buf381, (s0, 4096), (4096, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0), permute_785, out=buf391)
        del permute_785
        buf387 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (1024, s0), (1, 1024), 0), view_389, out=buf387)
        buf388 = reinterpret_tensor(buf389, (s0, 4096), (4096, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (s0, 1024), (1024, 1), 0), permute_780, out=buf388)
        del permute_780
        buf384 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (1024, s0), (1, 1024), 0), view_389, out=buf384)
        del view_389
        buf385 = reinterpret_tensor(buf380, (s0, 4096), (4096, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (s0, 1024), (1024, 1), 0), permute_775, out=buf385)
        del permute_775
        buf394 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf394, buf385, buf388, buf391, primals_177, add_5842, rsqrt_38, s0, 4096, stream=stream0)
        del primals_177
        buf392 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_81], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf385, buf388, buf391, add_5842, rsqrt_38, buf392, 4096, s0, stream=stream0)
        del add_5842
        del buf385
        del rsqrt_38
        buf395 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (4096, s0), (1, 4096), 0), view_387, out=buf395)
        del view_387
        buf396 = reinterpret_tensor(buf370, (s0, 14336), (14336, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (s0, 4096), (4096, 1), 0), permute_789, out=buf396)
        del permute_789
        buf397 = buf367; del buf367  # reuse
        buf400 = reinterpret_tensor(mm_131, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_131  # reuse
        # Topologically Sorted Source Nodes: [silu_18], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf400, buf396, mm_130, buf397, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf396
        del mm_130
        buf398 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (14336, s0), (1, 14336), 0), view_383, out=buf398)
        buf399 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (s0, 14336), (14336, 1), 0), permute_793, out=buf399)
        del permute_793
        buf401 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (14336, s0), (1, 14336), 0), view_383, out=buf401)
        del view_383
        buf402 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (s0, 14336), (14336, 1), 0), permute_798, out=buf402)
        del permute_798
        buf405 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_186], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf405, buf399, buf402, primals_173, add_5777, rsqrt_37, s0, 4096, stream=stream0)
        del primals_173
        buf403 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187, to_79], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf399, buf402, add_5777, rsqrt_37, buf403, 4096, s0, stream=stream0)
        del add_5777
        del buf399
        del rsqrt_37
        buf406 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_72, (s0, 4096), (4096, 1), 0), out=buf406)
        buf407 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (s0, 4096), (4096, 1), 0), permute_802, out=buf407)
        del permute_802
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf408 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf407, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_58, view_378, view_379, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_72, getitem_73, getitem_74, getitem_75, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_58
        del getitem_72
        del getitem_73
        del getitem_74
        del getitem_75
        del view_378
        del view_379
        buf409 = buf408[0]
        assert_size_stride(buf409, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf410 = buf408[1]
        assert_size_stride(buf410, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf411 = buf408[2]
        assert_size_stride(buf411, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf408
        buf419 = reinterpret_tensor(buf407, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf409, bmm, buf419, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf409
        buf412 = reinterpret_tensor(buf383, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf410, bmm, buf412, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf416 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf412, buf410, bmm, buf416, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf413 = reinterpret_tensor(buf412, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf411, buf413, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf420 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (4096, s0), (1, 4096), 0), view_369, out=buf420)
        buf421 = reinterpret_tensor(buf411, (s0, 4096), (4096, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (s0, 4096), (4096, 1), 0), permute_818, out=buf421)
        del permute_818
        buf417 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (1024, s0), (1, 1024), 0), view_369, out=buf417)
        buf418 = reinterpret_tensor(buf419, (s0, 4096), (4096, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (s0, 1024), (1024, 1), 0), permute_813, out=buf418)
        del permute_813
        buf414 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (1024, s0), (1, 1024), 0), view_369, out=buf414)
        del view_369
        buf415 = reinterpret_tensor(buf410, (s0, 4096), (4096, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (s0, 1024), (1024, 1), 0), permute_808, out=buf415)
        del permute_808
        buf424 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_180], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf424, buf415, buf418, buf421, primals_168, add_5539, rsqrt_36, s0, 4096, stream=stream0)
        del primals_168
        buf422 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_77], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf415, buf418, buf421, add_5539, rsqrt_36, buf422, 4096, s0, stream=stream0)
        del add_5539
        del buf415
        del rsqrt_36
        buf425 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (4096, s0), (1, 4096), 0), view_367, out=buf425)
        del view_367
        buf426 = reinterpret_tensor(buf400, (s0, 14336), (14336, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (s0, 4096), (4096, 1), 0), permute_822, out=buf426)
        del permute_822
        buf427 = buf397; del buf397  # reuse
        buf430 = reinterpret_tensor(mm_124, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_124  # reuse
        # Topologically Sorted Source Nodes: [silu_17], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf430, buf426, mm_123, buf427, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf426
        del mm_123
        buf428 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (14336, s0), (1, 14336), 0), view_363, out=buf428)
        buf429 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (s0, 14336), (14336, 1), 0), permute_826, out=buf429)
        del permute_826
        buf431 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf430, (14336, s0), (1, 14336), 0), view_363, out=buf431)
        del view_363
        buf432 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf430, (s0, 14336), (14336, 1), 0), permute_831, out=buf432)
        del permute_831
        buf435 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_176], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf435, buf429, buf432, primals_164, add_5474, rsqrt_35, s0, 4096, stream=stream0)
        del primals_164
        buf433 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_176, hidden_states_177, to_75], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf429, buf432, add_5474, rsqrt_35, buf433, 4096, s0, stream=stream0)
        del add_5474
        del buf429
        del rsqrt_35
        buf436 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_68, (s0, 4096), (4096, 1), 0), out=buf436)
        buf437 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (s0, 4096), (4096, 1), 0), permute_835, out=buf437)
        del permute_835
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf438 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf437, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_55, view_358, view_359, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_68, getitem_69, getitem_70, getitem_71, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_55
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_71
        del view_358
        del view_359
        buf439 = buf438[0]
        assert_size_stride(buf439, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf440 = buf438[1]
        assert_size_stride(buf440, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf441 = buf438[2]
        assert_size_stride(buf441, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf438
        buf449 = reinterpret_tensor(buf437, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf439, bmm, buf449, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf439
        buf442 = reinterpret_tensor(buf413, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf440, bmm, buf442, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf446 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf442, buf440, bmm, buf446, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf443 = reinterpret_tensor(buf442, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf441, buf443, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf450 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (4096, s0), (1, 4096), 0), view_349, out=buf450)
        buf451 = reinterpret_tensor(buf441, (s0, 4096), (4096, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (s0, 4096), (4096, 1), 0), permute_851, out=buf451)
        del permute_851
        buf447 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1024, s0), (1, 1024), 0), view_349, out=buf447)
        buf448 = reinterpret_tensor(buf449, (s0, 4096), (4096, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (s0, 1024), (1024, 1), 0), permute_846, out=buf448)
        del permute_846
        buf444 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (1024, s0), (1, 1024), 0), view_349, out=buf444)
        del view_349
        buf445 = reinterpret_tensor(buf440, (s0, 4096), (4096, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (s0, 1024), (1024, 1), 0), permute_841, out=buf445)
        del permute_841
        buf454 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_170], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf454, buf445, buf448, buf451, primals_159, add_5236, rsqrt_34, s0, 4096, stream=stream0)
        del primals_159
        buf452 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_73], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf445, buf448, buf451, add_5236, rsqrt_34, buf452, 4096, s0, stream=stream0)
        del add_5236
        del buf445
        del rsqrt_34
        buf455 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (4096, s0), (1, 4096), 0), view_347, out=buf455)
        del view_347
        buf456 = reinterpret_tensor(buf430, (s0, 14336), (14336, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (s0, 4096), (4096, 1), 0), permute_855, out=buf456)
        del permute_855
        buf457 = buf427; del buf427  # reuse
        buf460 = reinterpret_tensor(mm_117, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_117  # reuse
        # Topologically Sorted Source Nodes: [silu_16], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf460, buf456, mm_116, buf457, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf456
        del mm_116
        buf458 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (14336, s0), (1, 14336), 0), view_343, out=buf458)
        buf459 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (s0, 14336), (14336, 1), 0), permute_859, out=buf459)
        del permute_859
        buf461 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (14336, s0), (1, 14336), 0), view_343, out=buf461)
        del view_343
        buf462 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (s0, 14336), (14336, 1), 0), permute_864, out=buf462)
        del permute_864
        buf465 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_166], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf465, buf459, buf462, primals_155, add_5171, rsqrt_33, s0, 4096, stream=stream0)
        del primals_155
        buf463 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167, to_71], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf459, buf462, add_5171, rsqrt_33, buf463, 4096, s0, stream=stream0)
        del add_5171
        del buf459
        del rsqrt_33
        buf466 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_64, (s0, 4096), (4096, 1), 0), out=buf466)
        buf467 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (s0, 4096), (4096, 1), 0), permute_868, out=buf467)
        del permute_868
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf468 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf467, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_52, view_338, view_339, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_64, getitem_65, getitem_66, getitem_67, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_52
        del getitem_64
        del getitem_65
        del getitem_66
        del getitem_67
        del view_338
        del view_339
        buf469 = buf468[0]
        assert_size_stride(buf469, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf470 = buf468[1]
        assert_size_stride(buf470, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf471 = buf468[2]
        assert_size_stride(buf471, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf468
        buf479 = reinterpret_tensor(buf467, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf469, bmm, buf479, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf469
        buf472 = reinterpret_tensor(buf443, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf470, bmm, buf472, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf476 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf472, buf470, bmm, buf476, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf473 = reinterpret_tensor(buf472, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf471, buf473, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf480 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (4096, s0), (1, 4096), 0), view_329, out=buf480)
        buf481 = reinterpret_tensor(buf471, (s0, 4096), (4096, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (s0, 4096), (4096, 1), 0), permute_884, out=buf481)
        del permute_884
        buf477 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (1024, s0), (1, 1024), 0), view_329, out=buf477)
        buf478 = reinterpret_tensor(buf479, (s0, 4096), (4096, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (s0, 1024), (1024, 1), 0), permute_879, out=buf478)
        del permute_879
        buf474 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1024, s0), (1, 1024), 0), view_329, out=buf474)
        del view_329
        buf475 = reinterpret_tensor(buf470, (s0, 4096), (4096, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (s0, 1024), (1024, 1), 0), permute_874, out=buf475)
        del permute_874
        buf484 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_160], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf484, buf475, buf478, buf481, primals_150, add_4933, rsqrt_32, s0, 4096, stream=stream0)
        del primals_150
        buf482 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_69], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf475, buf478, buf481, add_4933, rsqrt_32, buf482, 4096, s0, stream=stream0)
        del add_4933
        del buf475
        del rsqrt_32
        buf485 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (4096, s0), (1, 4096), 0), view_327, out=buf485)
        del view_327
        buf486 = reinterpret_tensor(buf460, (s0, 14336), (14336, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (s0, 4096), (4096, 1), 0), permute_888, out=buf486)
        del permute_888
        buf487 = buf457; del buf457  # reuse
        buf490 = reinterpret_tensor(mm_110, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_110  # reuse
        # Topologically Sorted Source Nodes: [silu_15], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf490, buf486, mm_109, buf487, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf486
        del mm_109
        buf488 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (14336, s0), (1, 14336), 0), view_323, out=buf488)
        buf489 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (s0, 14336), (14336, 1), 0), permute_892, out=buf489)
        del permute_892
        buf491 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (14336, s0), (1, 14336), 0), view_323, out=buf491)
        del view_323
        buf492 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (s0, 14336), (14336, 1), 0), permute_897, out=buf492)
        del permute_897
        buf495 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_156], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf495, buf489, buf492, primals_146, add_4868, rsqrt_31, s0, 4096, stream=stream0)
        del primals_146
        buf493 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157, to_67], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf489, buf492, add_4868, rsqrt_31, buf493, 4096, s0, stream=stream0)
        del add_4868
        del buf489
        del rsqrt_31
        buf496 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_60, (s0, 4096), (4096, 1), 0), out=buf496)
        buf497 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (s0, 4096), (4096, 1), 0), permute_901, out=buf497)
        del permute_901
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf498 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf497, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_49, view_318, view_319, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_49
        del getitem_60
        del getitem_61
        del getitem_62
        del getitem_63
        del view_318
        del view_319
        buf499 = buf498[0]
        assert_size_stride(buf499, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf500 = buf498[1]
        assert_size_stride(buf500, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf501 = buf498[2]
        assert_size_stride(buf501, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf498
        buf509 = reinterpret_tensor(buf497, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf499, bmm, buf509, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf499
        buf502 = reinterpret_tensor(buf473, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf500, bmm, buf502, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf506 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf502, buf500, bmm, buf506, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf503 = reinterpret_tensor(buf502, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf501, buf503, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf510 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (4096, s0), (1, 4096), 0), view_309, out=buf510)
        buf511 = reinterpret_tensor(buf501, (s0, 4096), (4096, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0), permute_917, out=buf511)
        del permute_917
        buf507 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (1024, s0), (1, 1024), 0), view_309, out=buf507)
        buf508 = reinterpret_tensor(buf509, (s0, 4096), (4096, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (s0, 1024), (1024, 1), 0), permute_912, out=buf508)
        del permute_912
        buf504 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (1024, s0), (1, 1024), 0), view_309, out=buf504)
        del view_309
        buf505 = reinterpret_tensor(buf500, (s0, 4096), (4096, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (s0, 1024), (1024, 1), 0), permute_907, out=buf505)
        del permute_907
        buf514 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_150], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf514, buf505, buf508, buf511, primals_141, add_4630, rsqrt_30, s0, 4096, stream=stream0)
        del primals_141
        buf512 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_65], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf505, buf508, buf511, add_4630, rsqrt_30, buf512, 4096, s0, stream=stream0)
        del add_4630
        del buf505
        del rsqrt_30
        buf515 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (4096, s0), (1, 4096), 0), view_307, out=buf515)
        del view_307
        buf516 = reinterpret_tensor(buf490, (s0, 14336), (14336, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (s0, 4096), (4096, 1), 0), permute_921, out=buf516)
        del permute_921
        buf517 = buf487; del buf487  # reuse
        buf520 = reinterpret_tensor(mm_103, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_103  # reuse
        # Topologically Sorted Source Nodes: [silu_14], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf520, buf516, mm_102, buf517, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf516
        del mm_102
        buf518 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf517, (14336, s0), (1, 14336), 0), view_303, out=buf518)
        buf519 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf517, (s0, 14336), (14336, 1), 0), permute_925, out=buf519)
        del permute_925
        buf521 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (14336, s0), (1, 14336), 0), view_303, out=buf521)
        del view_303
        buf522 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (s0, 14336), (14336, 1), 0), permute_930, out=buf522)
        del permute_930
        buf525 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_146], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf525, buf519, buf522, primals_137, add_4565, rsqrt_29, s0, 4096, stream=stream0)
        del primals_137
        buf523 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147, to_63], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf519, buf522, add_4565, rsqrt_29, buf523, 4096, s0, stream=stream0)
        del add_4565
        del buf519
        del rsqrt_29
        buf526 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_56, (s0, 4096), (4096, 1), 0), out=buf526)
        buf527 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (s0, 4096), (4096, 1), 0), permute_934, out=buf527)
        del permute_934
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf528 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf527, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_46, view_298, view_299, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_56, getitem_57, getitem_58, getitem_59, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_46
        del getitem_56
        del getitem_57
        del getitem_58
        del getitem_59
        del view_298
        del view_299
        buf529 = buf528[0]
        assert_size_stride(buf529, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf530 = buf528[1]
        assert_size_stride(buf530, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf531 = buf528[2]
        assert_size_stride(buf531, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf528
        buf539 = reinterpret_tensor(buf527, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf529, bmm, buf539, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf529
        buf532 = reinterpret_tensor(buf503, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf530, bmm, buf532, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf536 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf532, buf530, bmm, buf536, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf533 = reinterpret_tensor(buf532, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf531, buf533, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf540 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf539, (4096, s0), (1, 4096), 0), view_289, out=buf540)
        buf541 = reinterpret_tensor(buf531, (s0, 4096), (4096, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf539, (s0, 4096), (4096, 1), 0), permute_950, out=buf541)
        del permute_950
        buf537 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (1024, s0), (1, 1024), 0), view_289, out=buf537)
        buf538 = reinterpret_tensor(buf539, (s0, 4096), (4096, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (s0, 1024), (1024, 1), 0), permute_945, out=buf538)
        del permute_945
        buf534 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (1024, s0), (1, 1024), 0), view_289, out=buf534)
        del view_289
        buf535 = reinterpret_tensor(buf530, (s0, 4096), (4096, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (s0, 1024), (1024, 1), 0), permute_940, out=buf535)
        del permute_940
        buf544 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_140], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf544, buf535, buf538, buf541, primals_132, add_4327, rsqrt_28, s0, 4096, stream=stream0)
        del primals_132
        buf542 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_61], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf535, buf538, buf541, add_4327, rsqrt_28, buf542, 4096, s0, stream=stream0)
        del add_4327
        del buf535
        del rsqrt_28
        buf545 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf544, (4096, s0), (1, 4096), 0), view_287, out=buf545)
        del view_287
        buf546 = reinterpret_tensor(buf520, (s0, 14336), (14336, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf544, (s0, 4096), (4096, 1), 0), permute_954, out=buf546)
        del permute_954
        buf547 = buf517; del buf517  # reuse
        buf550 = reinterpret_tensor(mm_96, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_96  # reuse
        # Topologically Sorted Source Nodes: [silu_13], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf550, buf546, mm_95, buf547, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf546
        del mm_95
        buf548 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf547, (14336, s0), (1, 14336), 0), view_283, out=buf548)
        buf549 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf547, (s0, 14336), (14336, 1), 0), permute_958, out=buf549)
        del permute_958
        buf551 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (14336, s0), (1, 14336), 0), view_283, out=buf551)
        del view_283
        buf552 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (s0, 14336), (14336, 1), 0), permute_963, out=buf552)
        del permute_963
        buf555 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf555, buf549, buf552, primals_128, add_4262, rsqrt_27, s0, 4096, stream=stream0)
        del primals_128
        buf553 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, to_59], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf549, buf552, add_4262, rsqrt_27, buf553, 4096, s0, stream=stream0)
        del add_4262
        del buf549
        del rsqrt_27
        buf556 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_52, (s0, 4096), (4096, 1), 0), out=buf556)
        buf557 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (s0, 4096), (4096, 1), 0), permute_967, out=buf557)
        del permute_967
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf558 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf557, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_43, view_278, view_279, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_52, getitem_53, getitem_54, getitem_55, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_43
        del getitem_52
        del getitem_53
        del getitem_54
        del getitem_55
        del view_278
        del view_279
        buf559 = buf558[0]
        assert_size_stride(buf559, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf560 = buf558[1]
        assert_size_stride(buf560, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf561 = buf558[2]
        assert_size_stride(buf561, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf558
        buf569 = reinterpret_tensor(buf557, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf559, bmm, buf569, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf559
        buf562 = reinterpret_tensor(buf533, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf560, bmm, buf562, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf566 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf562, buf560, bmm, buf566, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf563 = reinterpret_tensor(buf562, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf561, buf563, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf570 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (4096, s0), (1, 4096), 0), view_269, out=buf570)
        buf571 = reinterpret_tensor(buf561, (s0, 4096), (4096, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (s0, 4096), (4096, 1), 0), permute_983, out=buf571)
        del permute_983
        buf567 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (1024, s0), (1, 1024), 0), view_269, out=buf567)
        buf568 = reinterpret_tensor(buf569, (s0, 4096), (4096, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (s0, 1024), (1024, 1), 0), permute_978, out=buf568)
        del permute_978
        buf564 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (1024, s0), (1, 1024), 0), view_269, out=buf564)
        del view_269
        buf565 = reinterpret_tensor(buf560, (s0, 4096), (4096, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (s0, 1024), (1024, 1), 0), permute_973, out=buf565)
        del permute_973
        buf574 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_130], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf574, buf565, buf568, buf571, primals_123, add_4024, rsqrt_26, s0, 4096, stream=stream0)
        del primals_123
        buf572 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_57], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf565, buf568, buf571, add_4024, rsqrt_26, buf572, 4096, s0, stream=stream0)
        del add_4024
        del buf565
        del rsqrt_26
        buf575 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (4096, s0), (1, 4096), 0), view_267, out=buf575)
        del view_267
        buf576 = reinterpret_tensor(buf550, (s0, 14336), (14336, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (s0, 4096), (4096, 1), 0), permute_987, out=buf576)
        del permute_987
        buf577 = buf547; del buf547  # reuse
        buf580 = reinterpret_tensor(mm_89, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_89  # reuse
        # Topologically Sorted Source Nodes: [silu_12], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf580, buf576, mm_88, buf577, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf576
        del mm_88
        buf578 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (14336, s0), (1, 14336), 0), view_263, out=buf578)
        buf579 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (s0, 14336), (14336, 1), 0), permute_991, out=buf579)
        del permute_991
        buf581 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf580, (14336, s0), (1, 14336), 0), view_263, out=buf581)
        del view_263
        buf582 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf580, (s0, 14336), (14336, 1), 0), permute_996, out=buf582)
        del permute_996
        buf585 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf585, buf579, buf582, primals_119, add_3959, rsqrt_25, s0, 4096, stream=stream0)
        del primals_119
        buf583 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127, to_55], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf579, buf582, add_3959, rsqrt_25, buf583, 4096, s0, stream=stream0)
        del add_3959
        del buf579
        del rsqrt_25
        buf586 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_48, (s0, 4096), (4096, 1), 0), out=buf586)
        buf587 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (s0, 4096), (4096, 1), 0), permute_1000, out=buf587)
        del permute_1000
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf588 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf587, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_40, view_258, view_259, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_48, getitem_49, getitem_50, getitem_51, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_40
        del getitem_48
        del getitem_49
        del getitem_50
        del getitem_51
        del view_258
        del view_259
        buf589 = buf588[0]
        assert_size_stride(buf589, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf590 = buf588[1]
        assert_size_stride(buf590, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf591 = buf588[2]
        assert_size_stride(buf591, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf588
        buf599 = reinterpret_tensor(buf587, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf589, bmm, buf599, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf589
        buf592 = reinterpret_tensor(buf563, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf590, bmm, buf592, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf596 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf592, buf590, bmm, buf596, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf593 = reinterpret_tensor(buf592, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf591, buf593, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf600 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (4096, s0), (1, 4096), 0), view_249, out=buf600)
        buf601 = reinterpret_tensor(buf591, (s0, 4096), (4096, 1), 0); del buf591  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (s0, 4096), (4096, 1), 0), permute_1016, out=buf601)
        del permute_1016
        buf597 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (1024, s0), (1, 1024), 0), view_249, out=buf597)
        buf598 = reinterpret_tensor(buf599, (s0, 4096), (4096, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (s0, 1024), (1024, 1), 0), permute_1011, out=buf598)
        del permute_1011
        buf594 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (1024, s0), (1, 1024), 0), view_249, out=buf594)
        del view_249
        buf595 = reinterpret_tensor(buf590, (s0, 4096), (4096, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (s0, 1024), (1024, 1), 0), permute_1006, out=buf595)
        del permute_1006
        buf604 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_120], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf604, buf595, buf598, buf601, primals_114, add_3721, rsqrt_24, s0, 4096, stream=stream0)
        del primals_114
        buf602 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_53], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf595, buf598, buf601, add_3721, rsqrt_24, buf602, 4096, s0, stream=stream0)
        del add_3721
        del buf595
        del rsqrt_24
        buf605 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (4096, s0), (1, 4096), 0), view_247, out=buf605)
        del view_247
        buf606 = reinterpret_tensor(buf580, (s0, 14336), (14336, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (s0, 4096), (4096, 1), 0), permute_1020, out=buf606)
        del permute_1020
        buf607 = buf577; del buf577  # reuse
        buf610 = reinterpret_tensor(mm_82, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_82  # reuse
        # Topologically Sorted Source Nodes: [silu_11], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf610, buf606, mm_81, buf607, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf606
        del mm_81
        buf608 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (14336, s0), (1, 14336), 0), view_243, out=buf608)
        buf609 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (s0, 14336), (14336, 1), 0), permute_1024, out=buf609)
        del permute_1024
        buf611 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf610, (14336, s0), (1, 14336), 0), view_243, out=buf611)
        del view_243
        buf612 = buf598; del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf610, (s0, 14336), (14336, 1), 0), permute_1029, out=buf612)
        del permute_1029
        buf615 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_116], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf615, buf609, buf612, primals_110, add_3656, rsqrt_23, s0, 4096, stream=stream0)
        del primals_110
        buf613 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_117, to_51], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf609, buf612, add_3656, rsqrt_23, buf613, 4096, s0, stream=stream0)
        del add_3656
        del buf609
        del rsqrt_23
        buf616 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_44, (s0, 4096), (4096, 1), 0), out=buf616)
        buf617 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (s0, 4096), (4096, 1), 0), permute_1033, out=buf617)
        del permute_1033
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf618 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf617, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_37, view_238, view_239, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_44, getitem_45, getitem_46, getitem_47, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_37
        del getitem_44
        del getitem_45
        del getitem_46
        del getitem_47
        del view_238
        del view_239
        buf619 = buf618[0]
        assert_size_stride(buf619, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf620 = buf618[1]
        assert_size_stride(buf620, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf621 = buf618[2]
        assert_size_stride(buf621, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf618
        buf629 = reinterpret_tensor(buf617, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf619, bmm, buf629, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf619
        buf622 = reinterpret_tensor(buf593, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf620, bmm, buf622, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf626 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf622, buf620, bmm, buf626, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf623 = reinterpret_tensor(buf622, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf621, buf623, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf630 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (4096, s0), (1, 4096), 0), view_229, out=buf630)
        buf631 = reinterpret_tensor(buf621, (s0, 4096), (4096, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0), permute_1049, out=buf631)
        del permute_1049
        buf627 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf626, (1024, s0), (1, 1024), 0), view_229, out=buf627)
        buf628 = reinterpret_tensor(buf629, (s0, 4096), (4096, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf626, (s0, 1024), (1024, 1), 0), permute_1044, out=buf628)
        del permute_1044
        buf624 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf623, (1024, s0), (1, 1024), 0), view_229, out=buf624)
        del view_229
        buf625 = reinterpret_tensor(buf620, (s0, 4096), (4096, 1), 0); del buf620  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf623, (s0, 1024), (1024, 1), 0), permute_1039, out=buf625)
        del permute_1039
        buf634 = buf615; del buf615  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf634, buf625, buf628, buf631, primals_105, add_3418, rsqrt_22, s0, 4096, stream=stream0)
        del primals_105
        buf632 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_49], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf625, buf628, buf631, add_3418, rsqrt_22, buf632, 4096, s0, stream=stream0)
        del add_3418
        del buf625
        del rsqrt_22
        buf635 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (4096, s0), (1, 4096), 0), view_227, out=buf635)
        del view_227
        buf636 = reinterpret_tensor(buf610, (s0, 14336), (14336, 1), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (s0, 4096), (4096, 1), 0), permute_1053, out=buf636)
        del permute_1053
        buf637 = buf607; del buf607  # reuse
        buf640 = reinterpret_tensor(mm_75, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_75  # reuse
        # Topologically Sorted Source Nodes: [silu_10], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf640, buf636, mm_74, buf637, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf636
        del mm_74
        buf638 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (14336, s0), (1, 14336), 0), view_223, out=buf638)
        buf639 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (s0, 14336), (14336, 1), 0), permute_1057, out=buf639)
        del permute_1057
        buf641 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (14336, s0), (1, 14336), 0), view_223, out=buf641)
        del view_223
        buf642 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (s0, 14336), (14336, 1), 0), permute_1062, out=buf642)
        del permute_1062
        buf645 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf645, buf639, buf642, primals_101, add_3353, rsqrt_21, s0, 4096, stream=stream0)
        del primals_101
        buf643 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107, to_47], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf639, buf642, add_3353, rsqrt_21, buf643, 4096, s0, stream=stream0)
        del add_3353
        del buf639
        del rsqrt_21
        buf646 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf645, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_40, (s0, 4096), (4096, 1), 0), out=buf646)
        buf647 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf645, (s0, 4096), (4096, 1), 0), permute_1066, out=buf647)
        del permute_1066
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf648 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf647, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_34, view_218, view_219, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_40, getitem_41, getitem_42, getitem_43, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_34
        del getitem_40
        del getitem_41
        del getitem_42
        del getitem_43
        del view_218
        del view_219
        buf649 = buf648[0]
        assert_size_stride(buf649, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf650 = buf648[1]
        assert_size_stride(buf650, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf651 = buf648[2]
        assert_size_stride(buf651, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf648
        buf659 = reinterpret_tensor(buf647, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf647  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf649, bmm, buf659, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf649
        buf652 = reinterpret_tensor(buf623, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf650, bmm, buf652, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf656 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf652, buf650, bmm, buf656, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf653 = reinterpret_tensor(buf652, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf652  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf651, buf653, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf660 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (4096, s0), (1, 4096), 0), view_209, out=buf660)
        buf661 = reinterpret_tensor(buf651, (s0, 4096), (4096, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (s0, 4096), (4096, 1), 0), permute_1082, out=buf661)
        del permute_1082
        buf657 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (1024, s0), (1, 1024), 0), view_209, out=buf657)
        buf658 = reinterpret_tensor(buf659, (s0, 4096), (4096, 1), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (s0, 1024), (1024, 1), 0), permute_1077, out=buf658)
        del permute_1077
        buf654 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (1024, s0), (1, 1024), 0), view_209, out=buf654)
        del view_209
        buf655 = reinterpret_tensor(buf650, (s0, 4096), (4096, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (s0, 1024), (1024, 1), 0), permute_1072, out=buf655)
        del permute_1072
        buf664 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf664, buf655, buf658, buf661, primals_96, add_3115, rsqrt_20, s0, 4096, stream=stream0)
        del primals_96
        buf662 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_45], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf655, buf658, buf661, add_3115, rsqrt_20, buf662, 4096, s0, stream=stream0)
        del add_3115
        del buf655
        del rsqrt_20
        buf665 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (4096, s0), (1, 4096), 0), view_207, out=buf665)
        del view_207
        buf666 = reinterpret_tensor(buf640, (s0, 14336), (14336, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (s0, 4096), (4096, 1), 0), permute_1086, out=buf666)
        del permute_1086
        buf667 = buf637; del buf637  # reuse
        buf670 = reinterpret_tensor(mm_68, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_68  # reuse
        # Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf670, buf666, mm_67, buf667, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf666
        del mm_67
        buf668 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (14336, s0), (1, 14336), 0), view_203, out=buf668)
        buf669 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (s0, 14336), (14336, 1), 0), permute_1090, out=buf669)
        del permute_1090
        buf671 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf670, (14336, s0), (1, 14336), 0), view_203, out=buf671)
        del view_203
        buf672 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf670, (s0, 14336), (14336, 1), 0), permute_1095, out=buf672)
        del permute_1095
        buf675 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf675, buf669, buf672, primals_92, add_3050, rsqrt_19, s0, 4096, stream=stream0)
        del primals_92
        buf673 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97, to_43], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf669, buf672, add_3050, rsqrt_19, buf673, 4096, s0, stream=stream0)
        del add_3050
        del buf669
        del rsqrt_19
        buf676 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_36, (s0, 4096), (4096, 1), 0), out=buf676)
        buf677 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (s0, 4096), (4096, 1), 0), permute_1099, out=buf677)
        del permute_1099
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf678 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf677, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_31, view_198, view_199, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_36, getitem_37, getitem_38, getitem_39, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_31
        del getitem_36
        del getitem_37
        del getitem_38
        del getitem_39
        del view_198
        del view_199
        buf679 = buf678[0]
        assert_size_stride(buf679, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf680 = buf678[1]
        assert_size_stride(buf680, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf681 = buf678[2]
        assert_size_stride(buf681, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf678
        buf689 = reinterpret_tensor(buf677, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf679, bmm, buf689, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf679
        buf682 = reinterpret_tensor(buf653, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf653  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf680, bmm, buf682, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf686 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf682, buf680, bmm, buf686, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf683 = reinterpret_tensor(buf682, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf681, buf683, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf690 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (4096, s0), (1, 4096), 0), view_189, out=buf690)
        buf691 = reinterpret_tensor(buf681, (s0, 4096), (4096, 1), 0); del buf681  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (s0, 4096), (4096, 1), 0), permute_1115, out=buf691)
        del permute_1115
        buf687 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf686, (1024, s0), (1, 1024), 0), view_189, out=buf687)
        buf688 = reinterpret_tensor(buf689, (s0, 4096), (4096, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf686, (s0, 1024), (1024, 1), 0), permute_1110, out=buf688)
        del permute_1110
        buf684 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (1024, s0), (1, 1024), 0), view_189, out=buf684)
        del view_189
        buf685 = reinterpret_tensor(buf680, (s0, 4096), (4096, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (s0, 1024), (1024, 1), 0), permute_1105, out=buf685)
        del permute_1105
        buf694 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_90], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf694, buf685, buf688, buf691, primals_87, add_2812, rsqrt_18, s0, 4096, stream=stream0)
        del primals_87
        buf692 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_41], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf685, buf688, buf691, add_2812, rsqrt_18, buf692, 4096, s0, stream=stream0)
        del add_2812
        del buf685
        del rsqrt_18
        buf695 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf694, (4096, s0), (1, 4096), 0), view_187, out=buf695)
        del view_187
        buf696 = reinterpret_tensor(buf670, (s0, 14336), (14336, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf694, (s0, 4096), (4096, 1), 0), permute_1119, out=buf696)
        del permute_1119
        buf697 = buf667; del buf667  # reuse
        buf700 = reinterpret_tensor(mm_61, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_61  # reuse
        # Topologically Sorted Source Nodes: [silu_8], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf700, buf696, mm_60, buf697, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf696
        del mm_60
        buf698 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (14336, s0), (1, 14336), 0), view_183, out=buf698)
        buf699 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (s0, 14336), (14336, 1), 0), permute_1123, out=buf699)
        del permute_1123
        buf701 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (14336, s0), (1, 14336), 0), view_183, out=buf701)
        del view_183
        buf702 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (s0, 14336), (14336, 1), 0), permute_1128, out=buf702)
        del permute_1128
        buf705 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_86], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf705, buf699, buf702, primals_83, add_2747, rsqrt_17, s0, 4096, stream=stream0)
        del primals_83
        buf703 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87, to_39], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf699, buf702, add_2747, rsqrt_17, buf703, 4096, s0, stream=stream0)
        del add_2747
        del buf699
        del rsqrt_17
        buf706 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_32, (s0, 4096), (4096, 1), 0), out=buf706)
        buf707 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (s0, 4096), (4096, 1), 0), permute_1132, out=buf707)
        del permute_1132
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf708 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf707, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_28, view_178, view_179, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_32, getitem_33, getitem_34, getitem_35, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_28
        del getitem_32
        del getitem_33
        del getitem_34
        del getitem_35
        del view_178
        del view_179
        buf709 = buf708[0]
        assert_size_stride(buf709, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf710 = buf708[1]
        assert_size_stride(buf710, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf711 = buf708[2]
        assert_size_stride(buf711, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf708
        buf719 = reinterpret_tensor(buf707, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf709, bmm, buf719, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf709
        buf712 = reinterpret_tensor(buf683, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf710, bmm, buf712, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf716 = buf686; del buf686  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf712, buf710, bmm, buf716, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf713 = reinterpret_tensor(buf712, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf711, buf713, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf720 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (4096, s0), (1, 4096), 0), view_169, out=buf720)
        buf721 = reinterpret_tensor(buf711, (s0, 4096), (4096, 1), 0); del buf711  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (s0, 4096), (4096, 1), 0), permute_1148, out=buf721)
        del permute_1148
        buf717 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (1024, s0), (1, 1024), 0), view_169, out=buf717)
        buf718 = reinterpret_tensor(buf719, (s0, 4096), (4096, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (s0, 1024), (1024, 1), 0), permute_1143, out=buf718)
        del permute_1143
        buf714 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf713, (1024, s0), (1, 1024), 0), view_169, out=buf714)
        del view_169
        buf715 = reinterpret_tensor(buf710, (s0, 4096), (4096, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf713, (s0, 1024), (1024, 1), 0), permute_1138, out=buf715)
        del permute_1138
        buf724 = buf705; del buf705  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf724, buf715, buf718, buf721, primals_78, add_2509, rsqrt_16, s0, 4096, stream=stream0)
        del primals_78
        buf722 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_37], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf715, buf718, buf721, add_2509, rsqrt_16, buf722, 4096, s0, stream=stream0)
        del add_2509
        del buf715
        del rsqrt_16
        buf725 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (4096, s0), (1, 4096), 0), view_167, out=buf725)
        del view_167
        buf726 = reinterpret_tensor(buf700, (s0, 14336), (14336, 1), 0); del buf700  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (s0, 4096), (4096, 1), 0), permute_1152, out=buf726)
        del permute_1152
        buf727 = buf697; del buf697  # reuse
        buf730 = reinterpret_tensor(mm_54, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_54  # reuse
        # Topologically Sorted Source Nodes: [silu_7], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf730, buf726, mm_53, buf727, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf726
        del mm_53
        buf728 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (14336, s0), (1, 14336), 0), view_163, out=buf728)
        buf729 = buf721; del buf721  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (s0, 14336), (14336, 1), 0), permute_1156, out=buf729)
        del permute_1156
        buf731 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (14336, s0), (1, 14336), 0), view_163, out=buf731)
        del view_163
        buf732 = buf718; del buf718  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (s0, 14336), (14336, 1), 0), permute_1161, out=buf732)
        del permute_1161
        buf735 = buf724; del buf724  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf735, buf729, buf732, primals_74, add_2444, rsqrt_15, s0, 4096, stream=stream0)
        del primals_74
        buf733 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77, to_35], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf729, buf732, add_2444, rsqrt_15, buf733, 4096, s0, stream=stream0)
        del add_2444
        del buf729
        del rsqrt_15
        buf736 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_28, (s0, 4096), (4096, 1), 0), out=buf736)
        buf737 = buf732; del buf732  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (s0, 4096), (4096, 1), 0), permute_1165, out=buf737)
        del permute_1165
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf738 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf737, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_25, view_158, view_159, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_28, getitem_29, getitem_30, getitem_31, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_25
        del getitem_28
        del getitem_29
        del getitem_30
        del getitem_31
        del view_158
        del view_159
        buf739 = buf738[0]
        assert_size_stride(buf739, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf740 = buf738[1]
        assert_size_stride(buf740, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf741 = buf738[2]
        assert_size_stride(buf741, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf738
        buf749 = reinterpret_tensor(buf737, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf739, bmm, buf749, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf739
        buf742 = reinterpret_tensor(buf713, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf740, bmm, buf742, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf746 = buf716; del buf716  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf742, buf740, bmm, buf746, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf743 = reinterpret_tensor(buf742, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf741, buf743, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf750 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (4096, s0), (1, 4096), 0), view_149, out=buf750)
        buf751 = reinterpret_tensor(buf741, (s0, 4096), (4096, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0), permute_1181, out=buf751)
        del permute_1181
        buf747 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (1024, s0), (1, 1024), 0), view_149, out=buf747)
        buf748 = reinterpret_tensor(buf749, (s0, 4096), (4096, 1), 0); del buf749  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (s0, 1024), (1024, 1), 0), permute_1176, out=buf748)
        del permute_1176
        buf744 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (1024, s0), (1, 1024), 0), view_149, out=buf744)
        del view_149
        buf745 = reinterpret_tensor(buf740, (s0, 4096), (4096, 1), 0); del buf740  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (s0, 1024), (1024, 1), 0), permute_1171, out=buf745)
        del permute_1171
        buf754 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf754, buf745, buf748, buf751, primals_69, add_2206, rsqrt_14, s0, 4096, stream=stream0)
        del primals_69
        buf752 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_33], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf745, buf748, buf751, add_2206, rsqrt_14, buf752, 4096, s0, stream=stream0)
        del add_2206
        del buf745
        del rsqrt_14
        buf755 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (4096, s0), (1, 4096), 0), view_147, out=buf755)
        del view_147
        buf756 = reinterpret_tensor(buf730, (s0, 14336), (14336, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (s0, 4096), (4096, 1), 0), permute_1185, out=buf756)
        del permute_1185
        buf757 = buf727; del buf727  # reuse
        buf760 = reinterpret_tensor(mm_47, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_47  # reuse
        # Topologically Sorted Source Nodes: [silu_6], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf760, buf756, mm_46, buf757, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf756
        del mm_46
        buf758 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf757, (14336, s0), (1, 14336), 0), view_143, out=buf758)
        buf759 = buf751; del buf751  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf757, (s0, 14336), (14336, 1), 0), permute_1189, out=buf759)
        del permute_1189
        buf761 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (14336, s0), (1, 14336), 0), view_143, out=buf761)
        del view_143
        buf762 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (s0, 14336), (14336, 1), 0), permute_1194, out=buf762)
        del permute_1194
        buf765 = buf754; del buf754  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf765, buf759, buf762, primals_65, add_2141, rsqrt_13, s0, 4096, stream=stream0)
        del primals_65
        buf763 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, to_31], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf759, buf762, add_2141, rsqrt_13, buf763, 4096, s0, stream=stream0)
        del add_2141
        del buf759
        del rsqrt_13
        buf766 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_24, (s0, 4096), (4096, 1), 0), out=buf766)
        buf767 = buf762; del buf762  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (s0, 4096), (4096, 1), 0), permute_1198, out=buf767)
        del permute_1198
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf768 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf767, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_22, view_138, view_139, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_24, getitem_25, getitem_26, getitem_27, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_22
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_27
        del view_138
        del view_139
        buf769 = buf768[0]
        assert_size_stride(buf769, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf770 = buf768[1]
        assert_size_stride(buf770, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf771 = buf768[2]
        assert_size_stride(buf771, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf768
        buf779 = reinterpret_tensor(buf767, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf769, bmm, buf779, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf769
        buf772 = reinterpret_tensor(buf743, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf743  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf770, bmm, buf772, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf776 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf772, buf770, bmm, buf776, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf773 = reinterpret_tensor(buf772, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf772  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf771, buf773, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf780 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf779, (4096, s0), (1, 4096), 0), view_129, out=buf780)
        buf781 = reinterpret_tensor(buf771, (s0, 4096), (4096, 1), 0); del buf771  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf779, (s0, 4096), (4096, 1), 0), permute_1214, out=buf781)
        del permute_1214
        buf777 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (1024, s0), (1, 1024), 0), view_129, out=buf777)
        buf778 = reinterpret_tensor(buf779, (s0, 4096), (4096, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (s0, 1024), (1024, 1), 0), permute_1209, out=buf778)
        del permute_1209
        buf774 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf773, (1024, s0), (1, 1024), 0), view_129, out=buf774)
        del view_129
        buf775 = reinterpret_tensor(buf770, (s0, 4096), (4096, 1), 0); del buf770  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf773, (s0, 1024), (1024, 1), 0), permute_1204, out=buf775)
        del permute_1204
        buf784 = buf765; del buf765  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf784, buf775, buf778, buf781, primals_60, add_1903, rsqrt_12, s0, 4096, stream=stream0)
        del primals_60
        buf782 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_29], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf775, buf778, buf781, add_1903, rsqrt_12, buf782, 4096, s0, stream=stream0)
        del add_1903
        del buf775
        del rsqrt_12
        buf785 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (4096, s0), (1, 4096), 0), view_127, out=buf785)
        del view_127
        buf786 = reinterpret_tensor(buf760, (s0, 14336), (14336, 1), 0); del buf760  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (s0, 4096), (4096, 1), 0), permute_1218, out=buf786)
        del permute_1218
        buf787 = buf757; del buf757  # reuse
        buf790 = reinterpret_tensor(mm_40, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_40  # reuse
        # Topologically Sorted Source Nodes: [silu_5], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf790, buf786, mm_39, buf787, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf786
        del mm_39
        buf788 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (14336, s0), (1, 14336), 0), view_123, out=buf788)
        buf789 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (s0, 14336), (14336, 1), 0), permute_1222, out=buf789)
        del permute_1222
        buf791 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (14336, s0), (1, 14336), 0), view_123, out=buf791)
        del view_123
        buf792 = buf778; del buf778  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (s0, 14336), (14336, 1), 0), permute_1227, out=buf792)
        del permute_1227
        buf795 = buf784; del buf784  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf795, buf789, buf792, primals_56, add_1838, rsqrt_11, s0, 4096, stream=stream0)
        del primals_56
        buf793 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57, to_27], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf789, buf792, add_1838, rsqrt_11, buf793, 4096, s0, stream=stream0)
        del add_1838
        del buf789
        del rsqrt_11
        buf796 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_20, (s0, 4096), (4096, 1), 0), out=buf796)
        buf797 = buf792; del buf792  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (s0, 4096), (4096, 1), 0), permute_1231, out=buf797)
        del permute_1231
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf798 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf797, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_19, view_118, view_119, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_20, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_19
        del getitem_20
        del getitem_21
        del getitem_22
        del getitem_23
        del view_118
        del view_119
        buf799 = buf798[0]
        assert_size_stride(buf799, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf800 = buf798[1]
        assert_size_stride(buf800, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf801 = buf798[2]
        assert_size_stride(buf801, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf798
        buf809 = reinterpret_tensor(buf797, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf797  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf799, bmm, buf809, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf799
        buf802 = reinterpret_tensor(buf773, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf773  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf800, bmm, buf802, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf806 = buf776; del buf776  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf802, buf800, bmm, buf806, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf803 = reinterpret_tensor(buf802, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf802  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf801, buf803, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf810 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (4096, s0), (1, 4096), 0), view_109, out=buf810)
        buf811 = reinterpret_tensor(buf801, (s0, 4096), (4096, 1), 0); del buf801  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (s0, 4096), (4096, 1), 0), permute_1247, out=buf811)
        del permute_1247
        buf807 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (1024, s0), (1, 1024), 0), view_109, out=buf807)
        buf808 = reinterpret_tensor(buf809, (s0, 4096), (4096, 1), 0); del buf809  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (s0, 1024), (1024, 1), 0), permute_1242, out=buf808)
        del permute_1242
        buf804 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (1024, s0), (1, 1024), 0), view_109, out=buf804)
        del view_109
        buf805 = reinterpret_tensor(buf800, (s0, 4096), (4096, 1), 0); del buf800  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (s0, 1024), (1024, 1), 0), permute_1237, out=buf805)
        del permute_1237
        buf814 = buf795; del buf795  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf814, buf805, buf808, buf811, primals_51, add_1600, rsqrt_10, s0, 4096, stream=stream0)
        del primals_51
        buf812 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_25], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf805, buf808, buf811, add_1600, rsqrt_10, buf812, 4096, s0, stream=stream0)
        del add_1600
        del buf805
        del rsqrt_10
        buf815 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf814, (4096, s0), (1, 4096), 0), view_107, out=buf815)
        del view_107
        buf816 = reinterpret_tensor(buf790, (s0, 14336), (14336, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf814, (s0, 4096), (4096, 1), 0), permute_1251, out=buf816)
        del permute_1251
        buf817 = buf787; del buf787  # reuse
        buf820 = reinterpret_tensor(mm_33, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_33  # reuse
        # Topologically Sorted Source Nodes: [silu_4], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf820, buf816, mm_32, buf817, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf816
        del mm_32
        buf818 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf817, (14336, s0), (1, 14336), 0), view_103, out=buf818)
        buf819 = buf811; del buf811  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf817, (s0, 14336), (14336, 1), 0), permute_1255, out=buf819)
        del permute_1255
        buf821 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (14336, s0), (1, 14336), 0), view_103, out=buf821)
        del view_103
        buf822 = buf808; del buf808  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (s0, 14336), (14336, 1), 0), permute_1260, out=buf822)
        del permute_1260
        buf825 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf825, buf819, buf822, primals_47, add_1535, rsqrt_9, s0, 4096, stream=stream0)
        del primals_47
        buf823 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47, to_23], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf819, buf822, add_1535, rsqrt_9, buf823, 4096, s0, stream=stream0)
        del add_1535
        del buf819
        del rsqrt_9
        buf826 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_16, (s0, 4096), (4096, 1), 0), out=buf826)
        buf827 = buf822; del buf822  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (s0, 4096), (4096, 1), 0), permute_1264, out=buf827)
        del permute_1264
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf828 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf827, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_16, view_98, view_99, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_16
        del getitem_16
        del getitem_17
        del getitem_18
        del getitem_19
        del view_98
        del view_99
        buf829 = buf828[0]
        assert_size_stride(buf829, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf830 = buf828[1]
        assert_size_stride(buf830, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf831 = buf828[2]
        assert_size_stride(buf831, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf828
        buf839 = reinterpret_tensor(buf827, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf827  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf829, bmm, buf839, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf829
        buf832 = reinterpret_tensor(buf803, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf803  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf830, bmm, buf832, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf836 = buf806; del buf806  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf832, buf830, bmm, buf836, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf833 = reinterpret_tensor(buf832, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf832  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf831, buf833, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf840 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf839, (4096, s0), (1, 4096), 0), view_89, out=buf840)
        buf841 = reinterpret_tensor(buf831, (s0, 4096), (4096, 1), 0); del buf831  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf839, (s0, 4096), (4096, 1), 0), permute_1280, out=buf841)
        del permute_1280
        buf837 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf836, (1024, s0), (1, 1024), 0), view_89, out=buf837)
        buf838 = reinterpret_tensor(buf839, (s0, 4096), (4096, 1), 0); del buf839  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf836, (s0, 1024), (1024, 1), 0), permute_1275, out=buf838)
        del permute_1275
        buf834 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (1024, s0), (1, 1024), 0), view_89, out=buf834)
        del view_89
        buf835 = reinterpret_tensor(buf830, (s0, 4096), (4096, 1), 0); del buf830  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (s0, 1024), (1024, 1), 0), permute_1270, out=buf835)
        del permute_1270
        buf844 = buf825; del buf825  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf844, buf835, buf838, buf841, primals_42, add_1297, rsqrt_8, s0, 4096, stream=stream0)
        del primals_42
        buf842 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_21], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf835, buf838, buf841, add_1297, rsqrt_8, buf842, 4096, s0, stream=stream0)
        del add_1297
        del buf835
        del rsqrt_8
        buf845 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf844, (4096, s0), (1, 4096), 0), view_87, out=buf845)
        del view_87
        buf846 = reinterpret_tensor(buf820, (s0, 14336), (14336, 1), 0); del buf820  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf844, (s0, 4096), (4096, 1), 0), permute_1284, out=buf846)
        del permute_1284
        buf847 = buf817; del buf817  # reuse
        buf850 = reinterpret_tensor(mm_26, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_26  # reuse
        # Topologically Sorted Source Nodes: [silu_3], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf850, buf846, mm_25, buf847, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf846
        del mm_25
        buf848 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (14336, s0), (1, 14336), 0), view_83, out=buf848)
        buf849 = buf841; del buf841  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (s0, 14336), (14336, 1), 0), permute_1288, out=buf849)
        del permute_1288
        buf851 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (14336, s0), (1, 14336), 0), view_83, out=buf851)
        del view_83
        buf852 = buf838; del buf838  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (s0, 14336), (14336, 1), 0), permute_1293, out=buf852)
        del permute_1293
        buf855 = buf844; del buf844  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf855, buf849, buf852, primals_38, add_1232, rsqrt_7, s0, 4096, stream=stream0)
        del primals_38
        buf853 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37, to_19], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf849, buf852, add_1232, rsqrt_7, buf853, 4096, s0, stream=stream0)
        del add_1232
        del buf849
        del rsqrt_7
        buf856 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf855, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_12, (s0, 4096), (4096, 1), 0), out=buf856)
        buf857 = buf852; del buf852  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf855, (s0, 4096), (4096, 1), 0), permute_1297, out=buf857)
        del permute_1297
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf858 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf857, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_13, view_78, view_79, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_12, getitem_13, getitem_14, getitem_15, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_13
        del getitem_12
        del getitem_13
        del getitem_14
        del getitem_15
        del view_78
        del view_79
        buf859 = buf858[0]
        assert_size_stride(buf859, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf860 = buf858[1]
        assert_size_stride(buf860, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf861 = buf858[2]
        assert_size_stride(buf861, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf858
        buf869 = reinterpret_tensor(buf857, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf857  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf859, bmm, buf869, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf859
        buf862 = reinterpret_tensor(buf833, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf833  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf860, bmm, buf862, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf866 = buf836; del buf836  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf862, buf860, bmm, buf866, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf863 = reinterpret_tensor(buf862, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf862  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf861, buf863, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf870 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (4096, s0), (1, 4096), 0), view_69, out=buf870)
        buf871 = reinterpret_tensor(buf861, (s0, 4096), (4096, 1), 0); del buf861  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (s0, 4096), (4096, 1), 0), permute_1313, out=buf871)
        del permute_1313
        buf867 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf866, (1024, s0), (1, 1024), 0), view_69, out=buf867)
        buf868 = reinterpret_tensor(buf869, (s0, 4096), (4096, 1), 0); del buf869  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf866, (s0, 1024), (1024, 1), 0), permute_1308, out=buf868)
        del permute_1308
        buf864 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (1024, s0), (1, 1024), 0), view_69, out=buf864)
        del view_69
        buf865 = reinterpret_tensor(buf860, (s0, 4096), (4096, 1), 0); del buf860  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (s0, 1024), (1024, 1), 0), permute_1303, out=buf865)
        del permute_1303
        buf874 = buf855; del buf855  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_30], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf874, buf865, buf868, buf871, primals_33, add_994, rsqrt_6, s0, 4096, stream=stream0)
        del primals_33
        buf872 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_17], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf865, buf868, buf871, add_994, rsqrt_6, buf872, 4096, s0, stream=stream0)
        del add_994
        del buf865
        del rsqrt_6
        buf875 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf874, (4096, s0), (1, 4096), 0), view_67, out=buf875)
        del view_67
        buf876 = reinterpret_tensor(buf850, (s0, 14336), (14336, 1), 0); del buf850  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf874, (s0, 4096), (4096, 1), 0), permute_1317, out=buf876)
        del permute_1317
        buf877 = buf847; del buf847  # reuse
        buf880 = reinterpret_tensor(mm_19, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_19  # reuse
        # Topologically Sorted Source Nodes: [silu_2], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf880, buf876, mm_18, buf877, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf876
        del mm_18
        buf878 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf877, (14336, s0), (1, 14336), 0), view_63, out=buf878)
        buf879 = buf871; del buf871  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf877, (s0, 14336), (14336, 1), 0), permute_1321, out=buf879)
        del permute_1321
        buf881 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf880, (14336, s0), (1, 14336), 0), view_63, out=buf881)
        del view_63
        buf882 = buf868; del buf868  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf880, (s0, 14336), (14336, 1), 0), permute_1326, out=buf882)
        del permute_1326
        buf885 = buf874; del buf874  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf885, buf879, buf882, primals_29, add_929, rsqrt_5, s0, 4096, stream=stream0)
        del primals_29
        buf883 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27, to_15], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf879, buf882, add_929, rsqrt_5, buf883, 4096, s0, stream=stream0)
        del add_929
        del buf879
        del rsqrt_5
        buf886 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_8, (s0, 4096), (4096, 1), 0), out=buf886)
        buf887 = buf882; del buf882  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (s0, 4096), (4096, 1), 0), permute_1330, out=buf887)
        del permute_1330
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf888 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf887, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_10, view_58, view_59, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_8, getitem_9, getitem_10, getitem_11, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_10
        del getitem_10
        del getitem_11
        del getitem_8
        del getitem_9
        del view_58
        del view_59
        buf889 = buf888[0]
        assert_size_stride(buf889, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf890 = buf888[1]
        assert_size_stride(buf890, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf891 = buf888[2]
        assert_size_stride(buf891, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf888
        buf899 = reinterpret_tensor(buf887, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf887  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf889, bmm, buf899, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf889
        buf892 = reinterpret_tensor(buf863, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf863  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf890, bmm, buf892, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf896 = buf866; del buf866  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf892, buf890, bmm, buf896, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf893 = reinterpret_tensor(buf892, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf892  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf891, buf893, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf900 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf899, (4096, s0), (1, 4096), 0), view_49, out=buf900)
        buf901 = reinterpret_tensor(buf891, (s0, 4096), (4096, 1), 0); del buf891  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf899, (s0, 4096), (4096, 1), 0), permute_1346, out=buf901)
        del permute_1346
        buf897 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (1024, s0), (1, 1024), 0), view_49, out=buf897)
        buf898 = reinterpret_tensor(buf899, (s0, 4096), (4096, 1), 0); del buf899  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (s0, 1024), (1024, 1), 0), permute_1341, out=buf898)
        del permute_1341
        buf894 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf893, (1024, s0), (1, 1024), 0), view_49, out=buf894)
        del view_49
        buf895 = reinterpret_tensor(buf890, (s0, 4096), (4096, 1), 0); del buf890  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf893, (s0, 1024), (1024, 1), 0), permute_1336, out=buf895)
        del permute_1336
        buf904 = buf885; del buf885  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf904, buf895, buf898, buf901, primals_24, add_691, rsqrt_4, s0, 4096, stream=stream0)
        del primals_24
        buf902 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_13], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf895, buf898, buf901, add_691, rsqrt_4, buf902, 4096, s0, stream=stream0)
        del add_691
        del buf895
        del rsqrt_4
        buf905 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (4096, s0), (1, 4096), 0), view_47, out=buf905)
        del view_47
        buf906 = reinterpret_tensor(buf880, (s0, 14336), (14336, 1), 0); del buf880  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf904, (s0, 4096), (4096, 1), 0), permute_1350, out=buf906)
        del permute_1350
        buf907 = buf877; del buf877  # reuse
        buf910 = reinterpret_tensor(mm_12, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_12  # reuse
        # Topologically Sorted Source Nodes: [silu_1], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf910, buf906, mm_11, buf907, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf906
        del mm_11
        buf908 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (14336, s0), (1, 14336), 0), view_43, out=buf908)
        buf909 = buf901; del buf901  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (s0, 14336), (14336, 1), 0), permute_1354, out=buf909)
        del permute_1354
        buf911 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf910, (14336, s0), (1, 14336), 0), view_43, out=buf911)
        del view_43
        buf912 = buf898; del buf898  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf910, (s0, 14336), (14336, 1), 0), permute_1359, out=buf912)
        del permute_1359
        buf915 = buf904; del buf904  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf915, buf909, buf912, primals_20, add_626, rsqrt_3, s0, 4096, stream=stream0)
        del primals_20
        buf913 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17, to_11], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_5.run(buf909, buf912, add_626, rsqrt_3, buf913, 4096, s0, stream=stream0)
        del add_626
        del buf909
        del rsqrt_3
        buf916 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf915, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem_4, (s0, 4096), (4096, 1), 0), out=buf916)
        buf917 = buf912; del buf912  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf915, (s0, 4096), (4096, 1), 0), permute_1363, out=buf917)
        del permute_1363
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf918 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf917, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_7, view_38, view_39, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem_4, getitem_5, getitem_6, getitem_7, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_7
        del getitem_4
        del getitem_5
        del getitem_6
        del getitem_7
        del view_38
        del view_39
        buf919 = buf918[0]
        assert_size_stride(buf919, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf920 = buf918[1]
        assert_size_stride(buf920, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf921 = buf918[2]
        assert_size_stride(buf921, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf918
        buf929 = reinterpret_tensor(buf917, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf917  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf919, bmm, buf929, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del buf919
        buf922 = reinterpret_tensor(buf893, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf893  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf920, bmm, buf922, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf926 = buf896; del buf896  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf922, buf920, bmm, buf926, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        buf923 = reinterpret_tensor(buf922, (1, s0, 8, 128), (1024*s0, 1024, 128, 1), 0); del buf922  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf921, buf923, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf930 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf929, (4096, s0), (1, 4096), 0), view_29, out=buf930)
        buf931 = reinterpret_tensor(buf921, (s0, 4096), (4096, 1), 0); del buf921  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf929, (s0, 4096), (4096, 1), 0), permute_1379, out=buf931)
        del permute_1379
        buf927 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (1024, s0), (1, 1024), 0), view_29, out=buf927)
        buf928 = reinterpret_tensor(buf929, (s0, 4096), (4096, 1), 0); del buf929  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (s0, 1024), (1024, 1), 0), permute_1374, out=buf928)
        del permute_1374
        buf924 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf923, (1024, s0), (1, 1024), 0), view_29, out=buf924)
        del view_29
        buf925 = reinterpret_tensor(buf920, (s0, 4096), (4096, 1), 0); del buf920  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf923, (s0, 1024), (1024, 1), 0), permute_1369, out=buf925)
        del permute_1369
        buf934 = buf915; del buf915  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_10.run(buf934, buf925, buf928, buf931, primals_15, add_388, rsqrt_2, s0, 4096, stream=stream0)
        del primals_15
        buf932 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_9], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_11.run(buf925, buf928, buf931, add_388, rsqrt_2, buf932, 4096, s0, stream=stream0)
        del add_388
        del rsqrt_2
        buf935 = empty_strided_cuda((4096, 14336), (14336, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf934, (4096, s0), (1, 4096), 0), view_27, out=buf935)
        del view_27
        buf936 = reinterpret_tensor(buf910, (s0, 14336), (14336, 1), 0); del buf910  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf934, (s0, 4096), (4096, 1), 0), permute_1383, out=buf936)
        del permute_1383
        buf937 = buf907; del buf907  # reuse
        buf940 = reinterpret_tensor(mm_5, (1, s0, 14336), (14336*s0, 14336, 1), 0); del mm_5  # reuse
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel = 14336*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3.run(buf940, buf936, mm_4, buf937, triton_poi_fused_add_fill_mul_sigmoid_silu_sub_3_xnumel, stream=stream0)
        del buf936
        del mm_4
        buf938 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf937, (14336, s0), (1, 14336), 0), view_23, out=buf938)
        buf939 = buf931; del buf931  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf937, (s0, 14336), (14336, 1), 0), permute_1387, out=buf939)
        del buf937
        del permute_1387
        buf941 = empty_strided_cuda((14336, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf940, (14336, s0), (1, 14336), 0), view_23, out=buf941)
        del view_23
        buf942 = buf928; del buf928  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf940, (s0, 14336), (14336, 1), 0), permute_1392, out=buf942)
        del buf940
        del permute_1392
        buf945 = buf934; del buf934  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_12.run(buf945, buf939, buf942, primals_11, embedding, mm_3, rsqrt_1, s0, 4096, stream=stream0)
        del primals_11
        buf947 = buf925; del buf925  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (s0, 4096), (4096, 1), 0), permute_1396, out=buf947)
        del permute_1396
        buf946 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (4096, s0), (1, 4096), 0), reinterpret_tensor(getitem, (s0, 4096), (4096, 1), 0), out=buf946)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf948 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf947, (1, 32, s0, 128), (4096*s0, 128, 4096, 1), 0), clone_4, view_18, view_19, reinterpret_tensor(slice_19, (1, 32, s0, s0), (s0*s0 + 8*s0 + (-1)*s0*(s0 % 8), 0, 8 + s0 + (-1)*(s0 % 8), 1), 0), getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False], scale=0.08838834764831845)
        del clone_4
        del getitem
        del getitem_1
        del getitem_2
        del getitem_3
        del slice_19
        del view_18
        del view_19
        buf951 = buf948[2]
        assert_size_stride(buf951, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf950 = buf948[1]
        assert_size_stride(buf950, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        buf949 = buf948[0]
        assert_size_stride(buf949, (1, 32, s0, 128), (4096*s0, 128, 4096, 1))
        del buf948
        buf953 = buf923; del buf923  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf951, buf953, triton_poi_fused_clone_9_xnumel, stream=stream0)
        buf952 = reinterpret_tensor(buf926, (1, 8, s0, 128), (1024*s0, 128, 1024, 1), 0); del buf926  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_7.run(buf950, bmm, buf952, s0, triton_poi_fused_mul_7_xnumel, stream=stream0)
        buf956 = empty_strided_cuda((1, s0, 8, 128), (1024*s0, 1024, 128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf952, buf950, bmm, buf956, s0, triton_poi_fused_clone_8_xnumel, stream=stream0)
        del buf952
        buf959 = reinterpret_tensor(buf950, (1, s0, 32, 128), (4096*s0, 4096, 128, 1), 0); del buf950  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_6_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf949, bmm, buf959, s0, triton_poi_fused_clone_6_xnumel, stream=stream0)
        del bmm
        buf955 = reinterpret_tensor(buf949, (s0, 4096), (4096, 1), 0); del buf949  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf953, (s0, 1024), (1024, 1), 0), permute_1402, out=buf955)
        del permute_1402
        buf954 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf953, (1024, s0), (1, 1024), 0), view_9, out=buf954)
        del buf953
        buf958 = reinterpret_tensor(buf951, (s0, 4096), (4096, 1), 0); del buf951  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (s0, 1024), (1024, 1), 0), permute_1407, out=buf958)
        del permute_1407
        buf957 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (1024, s0), (1, 1024), 0), view_9, out=buf957)
        del buf956
        buf961 = buf947; del buf947  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (s0, 4096), (4096, 1), 0), permute_1412, out=buf961)
        del permute_1412
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.add, aten.mul, aten._to_copy, aten.sum, aten.div, aten.pow, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_13.run(buf955, buf958, buf961, primals_6, embedding, primals_2, buf945, rsqrt, buf965, s0, 4096, stream=stream0)
        del buf945
        del primals_2
        del primals_6
        buf943 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        buf962 = empty_strided_cuda((1, 1, 4096), (4096, 4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, hidden_states_7, to_7, hidden_states, hidden_states_1, to_5], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_sum_14.run(buf939, buf942, embedding, mm_3, rsqrt_1, buf955, buf958, buf961, rsqrt, buf943, buf962, 4096, s0, stream=stream0)
        del buf939
        del buf942
        del buf955
        del buf958
        del buf961
        del embedding
        del mm_3
        del rsqrt
        del rsqrt_1
        buf960 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (4096, s0), (1, 4096), 0), view_9, out=buf960)
        del buf959
        del view_9
        buf967 = empty_strided_cuda((128256, 4096), (4096, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_15.run(buf965, buf967, 525336576, stream=stream0)
        del buf965
    return (None, None, buf967, None, None, reinterpret_tensor(buf962, (4096, ), (1, ), 0), buf960, buf957, buf954, buf946, reinterpret_tensor(buf943, (4096, ), (1, ), 0), buf941, buf938, buf935, reinterpret_tensor(buf932, (4096, ), (1, ), 0), buf930, buf927, buf924, buf916, reinterpret_tensor(buf913, (4096, ), (1, ), 0), buf911, buf908, buf905, reinterpret_tensor(buf902, (4096, ), (1, ), 0), buf900, buf897, buf894, buf886, reinterpret_tensor(buf883, (4096, ), (1, ), 0), buf881, buf878, buf875, reinterpret_tensor(buf872, (4096, ), (1, ), 0), buf870, buf867, buf864, buf856, reinterpret_tensor(buf853, (4096, ), (1, ), 0), buf851, buf848, buf845, reinterpret_tensor(buf842, (4096, ), (1, ), 0), buf840, buf837, buf834, buf826, reinterpret_tensor(buf823, (4096, ), (1, ), 0), buf821, buf818, buf815, reinterpret_tensor(buf812, (4096, ), (1, ), 0), buf810, buf807, buf804, buf796, reinterpret_tensor(buf793, (4096, ), (1, ), 0), buf791, buf788, buf785, reinterpret_tensor(buf782, (4096, ), (1, ), 0), buf780, buf777, buf774, buf766, reinterpret_tensor(buf763, (4096, ), (1, ), 0), buf761, buf758, buf755, reinterpret_tensor(buf752, (4096, ), (1, ), 0), buf750, buf747, buf744, buf736, reinterpret_tensor(buf733, (4096, ), (1, ), 0), buf731, buf728, buf725, reinterpret_tensor(buf722, (4096, ), (1, ), 0), buf720, buf717, buf714, buf706, reinterpret_tensor(buf703, (4096, ), (1, ), 0), buf701, buf698, buf695, reinterpret_tensor(buf692, (4096, ), (1, ), 0), buf690, buf687, buf684, buf676, reinterpret_tensor(buf673, (4096, ), (1, ), 0), buf671, buf668, buf665, reinterpret_tensor(buf662, (4096, ), (1, ), 0), buf660, buf657, buf654, buf646, reinterpret_tensor(buf643, (4096, ), (1, ), 0), buf641, buf638, buf635, reinterpret_tensor(buf632, (4096, ), (1, ), 0), buf630, buf627, buf624, buf616, reinterpret_tensor(buf613, (4096, ), (1, ), 0), buf611, buf608, buf605, reinterpret_tensor(buf602, (4096, ), (1, ), 0), buf600, buf597, buf594, buf586, reinterpret_tensor(buf583, (4096, ), (1, ), 0), buf581, buf578, buf575, reinterpret_tensor(buf572, (4096, ), (1, ), 0), buf570, buf567, buf564, buf556, reinterpret_tensor(buf553, (4096, ), (1, ), 0), buf551, buf548, buf545, reinterpret_tensor(buf542, (4096, ), (1, ), 0), buf540, buf537, buf534, buf526, reinterpret_tensor(buf523, (4096, ), (1, ), 0), buf521, buf518, buf515, reinterpret_tensor(buf512, (4096, ), (1, ), 0), buf510, buf507, buf504, buf496, reinterpret_tensor(buf493, (4096, ), (1, ), 0), buf491, buf488, buf485, reinterpret_tensor(buf482, (4096, ), (1, ), 0), buf480, buf477, buf474, buf466, reinterpret_tensor(buf463, (4096, ), (1, ), 0), buf461, buf458, buf455, reinterpret_tensor(buf452, (4096, ), (1, ), 0), buf450, buf447, buf444, buf436, reinterpret_tensor(buf433, (4096, ), (1, ), 0), buf431, buf428, buf425, reinterpret_tensor(buf422, (4096, ), (1, ), 0), buf420, buf417, buf414, buf406, reinterpret_tensor(buf403, (4096, ), (1, ), 0), buf401, buf398, buf395, reinterpret_tensor(buf392, (4096, ), (1, ), 0), buf390, buf387, buf384, buf376, reinterpret_tensor(buf373, (4096, ), (1, ), 0), buf371, buf368, buf365, reinterpret_tensor(buf362, (4096, ), (1, ), 0), buf360, buf357, buf354, buf346, reinterpret_tensor(buf343, (4096, ), (1, ), 0), buf341, buf338, buf335, reinterpret_tensor(buf332, (4096, ), (1, ), 0), buf330, buf327, buf324, buf316, reinterpret_tensor(buf313, (4096, ), (1, ), 0), buf311, buf308, buf305, reinterpret_tensor(buf302, (4096, ), (1, ), 0), buf300, buf297, buf294, buf286, reinterpret_tensor(buf283, (4096, ), (1, ), 0), buf281, buf278, buf275, reinterpret_tensor(buf272, (4096, ), (1, ), 0), buf270, buf267, buf264, buf256, reinterpret_tensor(buf253, (4096, ), (1, ), 0), buf251, buf248, buf245, reinterpret_tensor(buf242, (4096, ), (1, ), 0), buf240, buf237, buf234, buf226, reinterpret_tensor(buf223, (4096, ), (1, ), 0), buf221, buf218, buf215, reinterpret_tensor(buf212, (4096, ), (1, ), 0), buf210, buf207, buf204, buf196, reinterpret_tensor(buf193, (4096, ), (1, ), 0), buf191, buf188, buf185, reinterpret_tensor(buf182, (4096, ), (1, ), 0), buf180, buf177, buf174, buf166, reinterpret_tensor(buf163, (4096, ), (1, ), 0), buf161, buf158, buf155, reinterpret_tensor(buf152, (4096, ), (1, ), 0), buf150, buf147, buf144, buf136, reinterpret_tensor(buf133, (4096, ), (1, ), 0), buf131, buf128, buf125, reinterpret_tensor(buf122, (4096, ), (1, ), 0), buf120, buf117, buf114, buf106, reinterpret_tensor(buf103, (4096, ), (1, ), 0), buf101, buf98, buf95, reinterpret_tensor(buf92, (4096, ), (1, ), 0), buf90, buf87, buf84, buf76, reinterpret_tensor(buf73, (4096, ), (1, ), 0), buf71, buf68, buf65, reinterpret_tensor(buf62, (4096, ), (1, ), 0), buf60, buf57, buf54, buf46, reinterpret_tensor(buf43, (4096, ), (1, ), 0), buf41, buf38, buf35, reinterpret_tensor(buf32, (4096, ), (1, ), 0), buf30, buf27, buf24, buf16, reinterpret_tensor(buf13, (4096, ), (1, ), 0), buf11, buf8, buf5, reinterpret_tensor(buf2, (4096, ), (1, ), 0), buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 4
    primals_2 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    primals_6 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_83 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_87 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_96 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_101 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_105 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_110 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_114 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_119 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_128 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_132 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_137 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_141 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_146 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_150 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_155 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_159 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_164 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_168 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_173 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_177 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_182 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_186 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_191 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_195 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_200 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_204 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_209 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_213 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_218 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_222 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_227 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_231 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_236 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_240 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_245 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_249 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_254 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_258 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_263 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_267 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_272 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_276 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_281 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_285 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_290 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_294 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    embedding = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    bmm = rand_strided((1, 64, 4), (256, 4, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_18 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_19 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_4 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    slice_19 = rand_strided((1, 1, 4, 4), (32, 32, 8, 1), device='cuda:0', dtype=torch.float16)
    getitem = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_1 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mm_3 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_1 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_4 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_5 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_27 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_388 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_2 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_38 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_39 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_7 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_4 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_5 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_626 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_3 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_11 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_12 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_47 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_691 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_4 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_58 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_59 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_10 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_8 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_9 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_929 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_5 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_18 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_19 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_67 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_994 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_6 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_78 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_79 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_13 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_12 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_13 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_1232 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_7 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_25 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_26 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_87 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_1297 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_8 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_98 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_99 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_16 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_16 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_17 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_1535 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_9 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_32 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_33 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_107 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_1600 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_10 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_118 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_119 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_19 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_20 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_21 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_23 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_1838 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_11 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_39 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_40 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_127 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_1903 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_12 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_138 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_139 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_22 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_24 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_25 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_2141 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_13 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_46 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_47 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_147 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_2206 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_14 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_158 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_159 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_25 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_28 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_29 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_31 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_2444 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_15 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_53 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_54 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_167 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_2509 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_16 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_178 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_179 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_28 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_32 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_33 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_35 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_2747 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_17 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_60 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_61 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_187 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_2812 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_18 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_189 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_198 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_199 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_31 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_36 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_37 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_3050 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_19 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_67 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_68 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_207 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_3115 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_20 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_209 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_218 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_219 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_34 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_40 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_41 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_43 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_3353 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_21 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_223 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_74 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_75 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_227 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_3418 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_22 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_229 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_238 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_239 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_37 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_44 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_45 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_47 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_3656 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_23 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_243 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_81 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_82 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_247 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_3721 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_24 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_249 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_258 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_259 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_40 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_48 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_49 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_3959 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_25 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_263 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_88 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_89 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_267 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_4024 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_26 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_269 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_278 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_279 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_43 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_52 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_53 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_55 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_4262 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_27 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_283 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_95 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_96 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_287 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_4327 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_28 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_289 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_298 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_299 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_46 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_56 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_57 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_4565 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_29 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_303 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_102 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_103 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_307 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_4630 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_30 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_309 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_318 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_319 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_49 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_60 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_61 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_4868 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_31 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_323 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_109 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_110 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_327 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_4933 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_32 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_338 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_339 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_52 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_64 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_65 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_67 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_5171 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_33 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_343 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_116 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_117 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_347 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_5236 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_34 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_349 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_358 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_359 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_55 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_68 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_69 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_71 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_5474 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_35 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_363 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_123 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_124 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_367 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_5539 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_36 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_369 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_378 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_379 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_58 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_72 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_73 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_5777 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_37 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_383 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_130 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_131 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_387 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_5842 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_38 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_389 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_398 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_399 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_61 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_76 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_77 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_79 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_6080 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_39 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_403 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_137 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_138 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_407 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_6145 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_40 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_409 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_418 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_419 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_64 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_80 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_81 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_82 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_83 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_6383 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_41 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_423 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_144 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_145 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_427 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_6448 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_42 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_429 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_438 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_439 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_67 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_84 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_85 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_6686 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_43 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_443 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_151 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_152 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_447 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_6751 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_44 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_449 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_458 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_459 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_70 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_88 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_89 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_91 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_6989 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_45 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_463 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_158 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_159 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_467 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_7054 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_46 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_469 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_478 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_479 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_73 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_92 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_93 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_7292 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_47 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_483 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_165 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_166 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_487 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_7357 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_48 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_489 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_498 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_499 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_76 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_96 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_97 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_98 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_7595 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_49 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_503 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_172 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_173 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_507 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_7660 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_50 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_509 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_518 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_519 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_79 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_100 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_101 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_103 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_7898 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_51 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_523 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_179 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_180 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_527 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_7963 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_52 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_529 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_538 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_539 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_82 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_104 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_105 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_107 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_8201 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_53 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_543 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_186 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_187 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_547 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_8266 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_54 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_549 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_558 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_559 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_85 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_108 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_109 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_8504 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_55 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_563 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_193 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_194 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_567 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_8569 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_56 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_569 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_578 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_579 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_88 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_112 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_113 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_115 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_8807 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_57 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_583 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_200 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_201 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_587 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_8872 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_58 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_589 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_598 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_599 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_91 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_116 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_117 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_119 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_9110 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_59 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_603 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_207 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_208 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_607 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_9175 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_60 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_609 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_618 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_619 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_94 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_120 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_121 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_122 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_9413 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_61 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_623 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_214 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_215 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_627 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_9478 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_62 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_629 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    view_638 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    view_639 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    clone_97 = rand_strided((1, 32, 4, 128), (16384, 512, 128, 1), device='cuda:0', dtype=torch.float16)
    getitem_124 = rand_strided((1, 32, 4, 128), (16384, 128, 4096, 1), device='cuda:0', dtype=torch.float16)
    getitem_125 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_127 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_9716 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_63 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_643 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    mm_221 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    mm_222 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    view_647 = rand_strided((4, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    add_9781 = rand_strided((1, 4, 4096), (16384, 4096, 1), device='cuda:0', dtype=torch.float16)
    rsqrt_64 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    view_649 = rand_strided((4, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_356 = rand_strided((128256, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_360 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_364 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_369 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_373 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_379 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_384 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_389 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_393 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_397 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_402 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_406 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_412 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_417 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_422 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_426 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_430 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_435 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_439 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_445 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_450 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_455 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_459 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_463 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_468 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_472 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_478 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_483 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_488 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_492 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_496 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_501 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_505 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_511 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_516 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_521 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_525 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_529 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_534 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_538 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_544 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_549 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_554 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_558 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_562 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_567 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_571 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_577 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_582 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_587 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_591 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_595 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_600 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_604 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_610 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_615 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_620 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_624 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_628 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_633 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_637 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_643 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_648 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_653 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_657 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_661 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_666 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_670 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_676 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_681 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_686 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_690 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_694 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_699 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_703 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_709 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_714 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_719 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_723 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_727 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_732 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_736 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_742 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_747 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_752 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_756 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_760 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_765 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_769 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_775 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_780 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_785 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_789 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_793 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_798 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_802 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_808 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_813 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_818 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_822 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_826 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_831 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_835 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_841 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_846 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_851 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_855 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_859 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_864 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_868 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_874 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_879 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_884 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_888 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_892 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_897 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_901 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_907 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_912 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_917 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_921 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_925 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_930 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_934 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_940 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_945 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_950 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_954 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_958 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_963 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_967 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_973 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_978 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_983 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_987 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_991 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_996 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1000 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1006 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1011 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1016 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1020 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1024 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1029 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1033 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1039 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1044 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1049 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1053 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1057 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1062 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1066 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1072 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1077 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1082 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1086 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1090 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1095 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1099 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1105 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1110 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1115 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1119 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1123 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1128 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1132 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1138 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1143 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1148 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1152 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1156 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1161 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1165 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1171 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1176 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1181 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1185 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1189 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1194 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1198 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1204 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1209 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1214 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1218 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1222 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1227 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1231 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1237 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1242 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1247 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1251 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1255 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1260 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1264 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1270 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1275 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1280 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1284 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1288 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1293 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1297 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1303 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1308 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1313 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1317 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1321 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1326 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1330 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1336 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1341 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1346 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1350 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1354 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1359 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1363 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1369 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1374 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1379 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1383 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.float16)
    permute_1387 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1392 = rand_strided((14336, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1396 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1402 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1407 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    permute_1412 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((1, 4, 128256), (513024, 128256, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_6, primals_11, primals_15, primals_20, primals_24, primals_29, primals_33, primals_38, primals_42, primals_47, primals_51, primals_56, primals_60, primals_65, primals_69, primals_74, primals_78, primals_83, primals_87, primals_92, primals_96, primals_101, primals_105, primals_110, primals_114, primals_119, primals_123, primals_128, primals_132, primals_137, primals_141, primals_146, primals_150, primals_155, primals_159, primals_164, primals_168, primals_173, primals_177, primals_182, primals_186, primals_191, primals_195, primals_200, primals_204, primals_209, primals_213, primals_218, primals_222, primals_227, primals_231, primals_236, primals_240, primals_245, primals_249, primals_254, primals_258, primals_263, primals_267, primals_272, primals_276, primals_281, primals_285, primals_290, primals_294, embedding, bmm, rsqrt, view_9, view_18, view_19, clone_4, slice_19, getitem, getitem_1, getitem_2, getitem_3, mm_3, rsqrt_1, view_23, mm_4, mm_5, view_27, add_388, rsqrt_2, view_29, view_38, view_39, clone_7, getitem_4, getitem_5, getitem_6, getitem_7, add_626, rsqrt_3, view_43, mm_11, mm_12, view_47, add_691, rsqrt_4, view_49, view_58, view_59, clone_10, getitem_8, getitem_9, getitem_10, getitem_11, add_929, rsqrt_5, view_63, mm_18, mm_19, view_67, add_994, rsqrt_6, view_69, view_78, view_79, clone_13, getitem_12, getitem_13, getitem_14, getitem_15, add_1232, rsqrt_7, view_83, mm_25, mm_26, view_87, add_1297, rsqrt_8, view_89, view_98, view_99, clone_16, getitem_16, getitem_17, getitem_18, getitem_19, add_1535, rsqrt_9, view_103, mm_32, mm_33, view_107, add_1600, rsqrt_10, view_109, view_118, view_119, clone_19, getitem_20, getitem_21, getitem_22, getitem_23, add_1838, rsqrt_11, view_123, mm_39, mm_40, view_127, add_1903, rsqrt_12, view_129, view_138, view_139, clone_22, getitem_24, getitem_25, getitem_26, getitem_27, add_2141, rsqrt_13, view_143, mm_46, mm_47, view_147, add_2206, rsqrt_14, view_149, view_158, view_159, clone_25, getitem_28, getitem_29, getitem_30, getitem_31, add_2444, rsqrt_15, view_163, mm_53, mm_54, view_167, add_2509, rsqrt_16, view_169, view_178, view_179, clone_28, getitem_32, getitem_33, getitem_34, getitem_35, add_2747, rsqrt_17, view_183, mm_60, mm_61, view_187, add_2812, rsqrt_18, view_189, view_198, view_199, clone_31, getitem_36, getitem_37, getitem_38, getitem_39, add_3050, rsqrt_19, view_203, mm_67, mm_68, view_207, add_3115, rsqrt_20, view_209, view_218, view_219, clone_34, getitem_40, getitem_41, getitem_42, getitem_43, add_3353, rsqrt_21, view_223, mm_74, mm_75, view_227, add_3418, rsqrt_22, view_229, view_238, view_239, clone_37, getitem_44, getitem_45, getitem_46, getitem_47, add_3656, rsqrt_23, view_243, mm_81, mm_82, view_247, add_3721, rsqrt_24, view_249, view_258, view_259, clone_40, getitem_48, getitem_49, getitem_50, getitem_51, add_3959, rsqrt_25, view_263, mm_88, mm_89, view_267, add_4024, rsqrt_26, view_269, view_278, view_279, clone_43, getitem_52, getitem_53, getitem_54, getitem_55, add_4262, rsqrt_27, view_283, mm_95, mm_96, view_287, add_4327, rsqrt_28, view_289, view_298, view_299, clone_46, getitem_56, getitem_57, getitem_58, getitem_59, add_4565, rsqrt_29, view_303, mm_102, mm_103, view_307, add_4630, rsqrt_30, view_309, view_318, view_319, clone_49, getitem_60, getitem_61, getitem_62, getitem_63, add_4868, rsqrt_31, view_323, mm_109, mm_110, view_327, add_4933, rsqrt_32, view_329, view_338, view_339, clone_52, getitem_64, getitem_65, getitem_66, getitem_67, add_5171, rsqrt_33, view_343, mm_116, mm_117, view_347, add_5236, rsqrt_34, view_349, view_358, view_359, clone_55, getitem_68, getitem_69, getitem_70, getitem_71, add_5474, rsqrt_35, view_363, mm_123, mm_124, view_367, add_5539, rsqrt_36, view_369, view_378, view_379, clone_58, getitem_72, getitem_73, getitem_74, getitem_75, add_5777, rsqrt_37, view_383, mm_130, mm_131, view_387, add_5842, rsqrt_38, view_389, view_398, view_399, clone_61, getitem_76, getitem_77, getitem_78, getitem_79, add_6080, rsqrt_39, view_403, mm_137, mm_138, view_407, add_6145, rsqrt_40, view_409, view_418, view_419, clone_64, getitem_80, getitem_81, getitem_82, getitem_83, add_6383, rsqrt_41, view_423, mm_144, mm_145, view_427, add_6448, rsqrt_42, view_429, view_438, view_439, clone_67, getitem_84, getitem_85, getitem_86, getitem_87, add_6686, rsqrt_43, view_443, mm_151, mm_152, view_447, add_6751, rsqrt_44, view_449, view_458, view_459, clone_70, getitem_88, getitem_89, getitem_90, getitem_91, add_6989, rsqrt_45, view_463, mm_158, mm_159, view_467, add_7054, rsqrt_46, view_469, view_478, view_479, clone_73, getitem_92, getitem_93, getitem_94, getitem_95, add_7292, rsqrt_47, view_483, mm_165, mm_166, view_487, add_7357, rsqrt_48, view_489, view_498, view_499, clone_76, getitem_96, getitem_97, getitem_98, getitem_99, add_7595, rsqrt_49, view_503, mm_172, mm_173, view_507, add_7660, rsqrt_50, view_509, view_518, view_519, clone_79, getitem_100, getitem_101, getitem_102, getitem_103, add_7898, rsqrt_51, view_523, mm_179, mm_180, view_527, add_7963, rsqrt_52, view_529, view_538, view_539, clone_82, getitem_104, getitem_105, getitem_106, getitem_107, add_8201, rsqrt_53, view_543, mm_186, mm_187, view_547, add_8266, rsqrt_54, view_549, view_558, view_559, clone_85, getitem_108, getitem_109, getitem_110, getitem_111, add_8504, rsqrt_55, view_563, mm_193, mm_194, view_567, add_8569, rsqrt_56, view_569, view_578, view_579, clone_88, getitem_112, getitem_113, getitem_114, getitem_115, add_8807, rsqrt_57, view_583, mm_200, mm_201, view_587, add_8872, rsqrt_58, view_589, view_598, view_599, clone_91, getitem_116, getitem_117, getitem_118, getitem_119, add_9110, rsqrt_59, view_603, mm_207, mm_208, view_607, add_9175, rsqrt_60, view_609, view_618, view_619, clone_94, getitem_120, getitem_121, getitem_122, getitem_123, add_9413, rsqrt_61, view_623, mm_214, mm_215, view_627, add_9478, rsqrt_62, view_629, view_638, view_639, clone_97, getitem_124, getitem_125, getitem_126, getitem_127, add_9716, rsqrt_63, view_643, mm_221, mm_222, view_647, add_9781, rsqrt_64, view_649, permute_356, permute_360, permute_364, permute_369, permute_373, permute_379, permute_384, permute_389, permute_393, permute_397, permute_402, permute_406, permute_412, permute_417, permute_422, permute_426, permute_430, permute_435, permute_439, permute_445, permute_450, permute_455, permute_459, permute_463, permute_468, permute_472, permute_478, permute_483, permute_488, permute_492, permute_496, permute_501, permute_505, permute_511, permute_516, permute_521, permute_525, permute_529, permute_534, permute_538, permute_544, permute_549, permute_554, permute_558, permute_562, permute_567, permute_571, permute_577, permute_582, permute_587, permute_591, permute_595, permute_600, permute_604, permute_610, permute_615, permute_620, permute_624, permute_628, permute_633, permute_637, permute_643, permute_648, permute_653, permute_657, permute_661, permute_666, permute_670, permute_676, permute_681, permute_686, permute_690, permute_694, permute_699, permute_703, permute_709, permute_714, permute_719, permute_723, permute_727, permute_732, permute_736, permute_742, permute_747, permute_752, permute_756, permute_760, permute_765, permute_769, permute_775, permute_780, permute_785, permute_789, permute_793, permute_798, permute_802, permute_808, permute_813, permute_818, permute_822, permute_826, permute_831, permute_835, permute_841, permute_846, permute_851, permute_855, permute_859, permute_864, permute_868, permute_874, permute_879, permute_884, permute_888, permute_892, permute_897, permute_901, permute_907, permute_912, permute_917, permute_921, permute_925, permute_930, permute_934, permute_940, permute_945, permute_950, permute_954, permute_958, permute_963, permute_967, permute_973, permute_978, permute_983, permute_987, permute_991, permute_996, permute_1000, permute_1006, permute_1011, permute_1016, permute_1020, permute_1024, permute_1029, permute_1033, permute_1039, permute_1044, permute_1049, permute_1053, permute_1057, permute_1062, permute_1066, permute_1072, permute_1077, permute_1082, permute_1086, permute_1090, permute_1095, permute_1099, permute_1105, permute_1110, permute_1115, permute_1119, permute_1123, permute_1128, permute_1132, permute_1138, permute_1143, permute_1148, permute_1152, permute_1156, permute_1161, permute_1165, permute_1171, permute_1176, permute_1181, permute_1185, permute_1189, permute_1194, permute_1198, permute_1204, permute_1209, permute_1214, permute_1218, permute_1222, permute_1227, permute_1231, permute_1237, permute_1242, permute_1247, permute_1251, permute_1255, permute_1260, permute_1264, permute_1270, permute_1275, permute_1280, permute_1284, permute_1288, permute_1293, permute_1297, permute_1303, permute_1308, permute_1313, permute_1317, permute_1321, permute_1326, permute_1330, permute_1336, permute_1341, permute_1346, permute_1350, permute_1354, permute_1359, permute_1363, permute_1369, permute_1374, permute_1379, permute_1383, permute_1387, permute_1392, permute_1396, permute_1402, permute_1407, permute_1412, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
