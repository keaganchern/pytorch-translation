import triton
import triton.language as tl
import torch
from triton.language.extra.libdevice import tanh


@triton.jit
def geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)

def pytorch_geglu_tanh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a * a * a
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * a_cubed)
    tanh_result = torch.tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)
    c = geglu_a * b
    return c

def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pass

def geglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    geglu_tanh_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        # num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)



def generate_random_tensor(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")

def run_test_case(shape):
    a = generate_random_tensor(shape)
    b = generate_random_tensor(shape)
    a_return, b_return, c_return = geglu_forward(a, b)
    return a_return, b_return, c_return


# Test cases
test_cases = [
    (1, 16),    # small
    (4, 1024),  # medium
    (16, 8192),  # large
]


results = []
for shape in test_cases:
    a, b, c = run_test_case(shape)
    results.append((shape, a, b, c))

opt_geglu = torch.compile(pytorch_geglu_tanh)

for shape, a, b, c in results:
    a_torch = a.clone().detach()
    b_torch = b.clone().detach()

    c_torch = opt_geglu(a_torch, b_torch)
    torch.testing.assert_close(c, c_torch, atol=1e-2, rtol=1e-2)