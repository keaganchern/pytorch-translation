#source
import torch
import triton
import triton.language as tl

from triton.language.extra.libdevice import rsqrt
@triton.jit
def test(
    x_ptr,
    y_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INCLUDE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        ax = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        ay = tl.load(
            y_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        a = ax + ay
        tl.store(x_ptr + cols, a, mask=mask)
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

#translation
def test_pytorch(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float, stride: int, INCLUDE_WEIGHT: bool) -> tuple[torch.Tensor, torch.Tensor]:
    a = x + y
    
    n_cols = a.shape[-1]
    sum_a_sq = torch.sum(a * a, dim=-1, keepdim=True)

    mean_of_squares = sum_a_sq / n_cols
    rstd = torch.rsqrt(mean_of_squares + eps)

 
    if INCLUDE_WEIGHT:
        h1 = a * rstd * w
    else:
        h1 = a * rstd

    return a, h1

def _rms_norm_add_forward(x, y, attn_norm_weights, eps):
    # x, y contiguous of same shape [..., n]
    # output of same shape, normed over the last dim.
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous")
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError("weights must be contiguous")
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    y_arg = y.reshape(-1, x.shape[-1])
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
            y_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            INCLUDE_WEIGHT=True,
        )
    return out


# Assume `_rms_norm_add_forward` is available in the namespace.

def test_rms_norm_add_small():
    # Small shape: batch 2, features 64
    x = torch.randn(2, 64, device='cuda', dtype=torch.float32)
    y = torch.randn(2, 64, device='cuda', dtype=torch.float32)
    w = torch.randn(64, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_add_forward(x, y, w, eps)
    return x, y, w, out

def test_rms_norm_add_medium():
    # Medium shape: batch 8, features 256
    x = torch.randn(8, 256, device='cuda', dtype=torch.float32)
    y = torch.randn(8, 256, device='cuda', dtype=torch.float32)
    w = torch.randn(256, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_add_forward(x, y, w, eps)
    return x, y, w, out

def test_rms_norm_add_large():
    # Large shape: batch 16, features 1024
    x = torch.randn(16, 1024, device='cuda', dtype=torch.float32)
    y = torch.randn(16, 1024, device='cuda', dtype=torch.float32)
    w = torch.randn(1024, device='cuda', dtype=torch.float32)
    eps = 1e-5
    out = _rms_norm_add_forward(x, y, w, eps)
    return x, y, w, out

# Execute test cases
x_small, y_small, w_small, out_small = test_rms_norm_add_small()
x_medium, y_medium, w_medium, out_medium = test_rms_norm_add_medium()
x_large, y_large, w_large, out_large = test_rms_norm_add_large()

opt_test = torch.compile(test_pytorch)

_, h1_small = test_pytorch(x_small, y_small, w_small, 1e-5, False)
torch.testing.assert_close(h1_small, out_small, rtol=1e-5, atol=1e-5)


_, h1_medium = test_pytorch(x_medium, y_medium, w_medium, 1e-5, False)
torch.testing.assert_close(h1_medium, out_medium, rtol=1e-5, atol=1e-5)

_, h1_large = test_pytorch(x_large, y_large, w_large, 1e-5, False)
torch.testing.assert_close(h1_large, out_large, rtol=1e-5, atol=1e-5)