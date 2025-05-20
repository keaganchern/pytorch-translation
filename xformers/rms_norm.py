import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt




@triton.jit
def test(
    x_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INCLUDE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + cols, mask=cols < N_COLS, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _mean += a * a
    rstd = rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)



def test_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float, include_weight: bool) -> torch.Tensor:
    """
    Rewrites the Triton RMSNorm kernel in PyTorch.

    Args:
        x: Input tensor. The normalization is applied over the last dimension.
        weight: Weight tensor for scaling. Should be broadcastable to x's shape
                after normalization, typically 1D matching x's last dimension.
        eps: A small float value added to the variance for numerical stability.
        include_weight: A boolean indicating whether to apply the weight.

    Returns:
        The normalized and optionally weighted tensor.
    """
    x_float = x.float()

    
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)

    
    rstd = torch.rsqrt(variance + eps)


    normalized_x = x_float * rstd

    if include_weight:
      
        output = normalized_x * weight
    else:
        output = normalized_x

    return output

def _rms_norm_forward(x, attn_norm_weights, eps):
    if not x.is_contiguous():
        raise ValueError("data must be contiguous")
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError("weights must be contiguous")
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        test[(M,)](
            x_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            INCLUDE_WEIGHT=attn_norm_weights is not None,
        )
    return out

def test_rms_norm_small():
    x = torch.randn(8, 128, device='cuda', dtype=torch.float32)
    w = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_forward(x, w, eps)
    return x, w, out

def test_rms_norm_medium():
    x = torch.randn(16, 1024, device='cuda', dtype=torch.float32)
    w = torch.randn(1024, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_forward(x, w, eps)
    return x, w, out

def test_rms_norm_large():
    x = torch.randn(32, 4096, device='cuda', dtype=torch.float32)
    w = torch.randn(4096, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_forward(x, w, eps)
    return x, w, out


small_x, small_w, small_out = test_rms_norm_small()
medium_x, medium_w, medium_out = test_rms_norm_medium()
large_x, large_w, large_out  = test_rms_norm_large()

opt_test = torch.compile(test_pytorch)

assert torch.allclose(small_out, opt_test(small_x, small_w, 1e-5, True), atol=1e-6, rtol=1e-4)

assert torch.allclose(medium_out, opt_test(medium_x, medium_w, 1e-5, True), atol=1e-6, rtol=1e-4)

assert torch.allclose(large_out, opt_test(large_x, large_w, 1e-5, True), atol=1e-6, rtol=1e-4)