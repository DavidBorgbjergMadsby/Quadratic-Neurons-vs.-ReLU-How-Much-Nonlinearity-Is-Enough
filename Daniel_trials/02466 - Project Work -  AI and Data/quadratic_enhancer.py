import triton
import triton.language as tl
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# Configurations for block sizes.
def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 256}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_D": 256}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_D": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 32, "BLOCK_SIZE_D": 64}, num_stages=5, num_warps=2
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 256}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 256, "BLOCK_SIZE_D": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 256, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_D": 256}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 128, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4
        ),
    ]


# turn on auto-tuning of block sizes for forward kernel
# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['B', 'D'],
# )


@triton.jit
def quadEnhancer_forward_kernel(
    x_ptr,
    w_ptr,
    k,
    output_ptr,
    B,
    D,
    stride_B,
    stride_D,
    BLOCK_SIZE_B: tl.constexpr = 32,
    BLOCK_SIZE_D: tl.constexpr = 64,
):
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)

    block_start_b = pid_b * BLOCK_SIZE_B  # column ID
    block_start_d = pid_d * BLOCK_SIZE_D  # row ID

    base_arr_b = tl.arange(0, BLOCK_SIZE_B)
    base_arr_d = tl.arange(0, BLOCK_SIZE_D)
    offsets_b = block_start_b + base_arr_b
    offsets_d = block_start_d + base_arr_d

    offsets = offsets_b[:, None] * stride_B + offsets_d[None, :] * stride_D
    ptrs = x_ptr + offsets
    mask = (offsets_b < B)[:, None] & (offsets_d < D)[None, :]

    y = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_D), dtype=tl.float32) + 1
    x = tl.load(ptrs, mask=mask, other=0.0)
    xType = x.dtype
    for i in range(1, k + 1):
        y_off_d = block_start_d + base_arr_d + i
        if block_start_d + BLOCK_SIZE_D + i >= D:
            y_mask = (y_off_d >= D) & (y_off_d < D + i)
            y_off_d = y_off_d - D * y_mask
        y_offsets = offsets_b[:, None] * stride_B + y_off_d[None, :] * stride_D
        y_ptrs = x_ptr + y_offsets
        y_mask = (offsets_b < B)[:, None] & (y_off_d < D)[None, :]
        w_off = base_arr_d + block_start_d
        w_mask = w_off < D
        w_off = (i - 1) * D + w_off
        y = (
            y
            + tl.load(y_ptrs, mask=y_mask, other=0.0)
            * tl.load(w_ptr + w_off, mask=w_mask, other=0.0)[None, :]
        )
    y = x * y
    y = y.to(xType)
    tl.store(output_ptr + offsets, y, mask=mask)


# turn on auto-tuning of block sizes for backward kernel
# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['B', 'D'],
#     reset_to_zero=['grad_w_ptr']
# )
@triton.jit
def quadEnhancer_backward_kernel(
    x_ptr,
    w_ptr,
    grad_z_ptr,
    k,
    grad_x_ptr,
    grad_w_ptr,
    B,
    D,
    BLOCK_SIZE_B: tl.constexpr = 32,
    BLOCK_SIZE_D: tl.constexpr = 64,
):
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    block_start_b = pid_b * BLOCK_SIZE_B  # column ID
    block_start_d = pid_d * BLOCK_SIZE_D  # row ID
    base_arr_b = tl.arange(0, BLOCK_SIZE_B)
    base_arr_d = tl.arange(0, BLOCK_SIZE_D)
    offsets_b = block_start_b + base_arr_b
    offsets_d = block_start_d + base_arr_d

    offsets = offsets_b[:, None] * D + offsets_d[None, :]
    x_ptrs = x_ptr + offsets
    grad_z_ptrs = grad_z_ptr + offsets
    mask = (offsets_b < B)[:, None] & (offsets_d < D)[None, :]

    grad_x = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_D), dtype=tl.float32)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    grad_z = tl.load(grad_z_ptrs, mask=mask, other=0.0)
    grad_x += grad_z
    for i in range(1, k + 1):
        y_col_pos_ind = block_start_d + base_arr_d + i  # column index
        y_col_neg_ind = block_start_d + base_arr_d - i
        if block_start_d + BLOCK_SIZE_D + i >= D:
            y_pos_mask = (y_col_pos_ind >= D) & (y_col_pos_ind < D + i)
            y_col_pos_ind = y_col_pos_ind - D * y_pos_mask
        if block_start_d - i < 0:
            y_neg_mask = y_col_neg_ind < 0
            y_col_neg_ind = y_col_neg_ind + D * y_neg_mask

        y_pos_offsets = offsets_b[:, None] * D + y_col_pos_ind[None, :]
        y_neg_offsets = offsets_b[:, None] * D + y_col_neg_ind[None, :]
        y_pos_ptrs = x_ptr + y_pos_offsets
        y_neg_ptrs = x_ptr + y_neg_offsets
        y_pos_mask = (offsets_b < B)[:, None] & (y_col_pos_ind < D)[None, :]
        y_neg_mask = (offsets_b < B)[:, None] & (y_col_neg_ind < D)[None, :]
        y_pos = tl.load(y_pos_ptrs, mask=y_pos_mask, other=0.0)
        y_neg = tl.load(y_neg_ptrs, mask=y_neg_mask, other=0.0)

        w_grad = grad_z * x * y_pos
        w_grad = tl.sum(w_grad, axis=0)
        w_ind = base_arr_d + block_start_d
        w_neg_ind = base_arr_d + block_start_d - i
        if block_start_d - i < 0:
            w_neg_mask = w_neg_ind < 0
            w_neg_ind = w_neg_ind + D * w_neg_mask
        w_mask = w_ind < D
        w_neg_mask = w_neg_ind < D
        w_ind = (i - 1) * D + w_ind
        w_neg_ind = (i - 1) * D + w_neg_ind
        tl.atomic_add(grad_w_ptr + w_ind, w_grad, mask=w_mask)
        grad_z_neg = tl.load(grad_z_ptr + y_neg_offsets, mask=y_neg_mask, other=0.0)
        w = tl.load(w_ptr + w_ind, mask=w_mask, other=0.0)
        w_neg = tl.load(w_ptr + w_neg_ind, mask=w_neg_mask, other=0.0)
        grad_x += w_neg * grad_z_neg * y_neg + grad_z * w * y_pos

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


def quadEnhancer_forward(x: torch.Tensor, w: torch.Tensor):
    """
    z = \Lambda @ x * x + x
    :param x: input tensor of shape [B,D]
    :param w: weight tensor of shape [k,D]
    :return: output tensor of shape [B,D]
    """
    assert x.device == w.device
    z = torch.empty_like(x, device=x.device)
    assert x.dim() == 2
    assert w.dim() == 2
    assert x.is_contiguous() and w.is_contiguous()
    k = w.shape[0]
    B, D = x.shape
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_SIZE_B"]),
        triton.cdiv(D, meta["BLOCK_SIZE_D"]),
    )
    quadEnhancer_forward_kernel[grid](x, w, k, z, B, D, x.stride(0), x.stride(1))
    return z


def quadEnhancer_backward(x: torch.Tensor, w: torch.Tensor, grad_z: torch.Tensor):
    """
    calculate gradients
    :param x: input tensor of shape [B,D]
    :param w: weight tensor of shape [k,D]
    :param z: gradient tensor of shape [B,D]
    :return: gradients of tensor x and tensor w
    """
    assert x.device == w.device
    assert x.dim() == 2
    assert w.dim() == 2
    assert x.is_contiguous() and w.is_contiguous() and grad_z.is_contiguous()

    grad_w = torch.zeros_like(w, device=x.device)
    grad_x = torch.zeros_like(x, device=x.device)

    k = w.shape[0]
    B, D = x.shape

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_SIZE_B"]),
        triton.cdiv(D, meta["BLOCK_SIZE_D"]),
    )
    quadEnhancer_backward_kernel[grid](x, w, grad_z, k, grad_x, grad_w, B, D)
    return grad_x, grad_w


class QuadEnhancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        z = quadEnhancer_forward(x, w)
        return z

    def backward(ctx, z_grad):
        x, w = ctx.saved_tensors
        grad_x, grad_w = quadEnhancer_backward(x, w, z_grad.contiguous())
        return grad_x, grad_w


class QuadEnhancer(nn.Module):
    def __init__(self, k, d):
        super(QuadEnhancer, self).__init__()
        self.d = d
        self.weight = nn.Parameter(torch.empty(k, d))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return QuadEnhancerFunction.apply(x, self.weight)

    def extra_repr(self) -> str:
        return f"hidden_features={self.d}, k={self.k}"


class QuadEnhancedLinear(nn.Module):
    static_k = -1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        k = k if self.static_k < 0 else self.static_k
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.qe = QuadEnhancer(k, self.out_features)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input = F.linear(input, self.weight)
        input = self.qe(input)
        return input + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, k={self.k}"
