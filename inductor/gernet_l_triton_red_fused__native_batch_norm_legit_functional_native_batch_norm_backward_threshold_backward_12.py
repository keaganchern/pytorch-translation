import triton
import triton.language as tl
import torch

@triton.jit
def test(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1920
    x1 = (xindex // 1920)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1920*r2) + (491520*x1)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (1920*r2) + (491520*x1)), rmask, other=0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (1920*r2) + (491520*x1)), rmask, other=0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)

# Constants
xnumel = 61440
rnumel = 256
XBLOCK = 256
RBLOCK = 256

# Calculate grid dimensions
grid_x = (xnumel + XBLOCK - 1) // XBLOCK
grid = (grid_x,)

# Sizes
size_in_ptr = 15_728_640  # For in_ptr0, in_ptr1, in_ptr2
size_in_ptr3 = 1920
size_out_ptr = xnumel  # For out_ptr0, out_ptr1

device = torch.device('cuda')

# Allocate input tensors with random values
in_ptr0 = torch.randn(size_in_ptr, device=device, dtype=torch.float32)
in_ptr0_copy = in_ptr0.clone()
in_ptr1 = torch.randn(size_in_ptr, device=device, dtype=torch.float32)
in_ptr1_copy = in_ptr1.clone()
in_ptr2 = torch.randn(size_in_ptr, device=device, dtype=torch.float32)
in_ptr2_copy = in_ptr2.clone()
in_ptr3 = torch.randn(size_in_ptr3, device=device, dtype=torch.float32)
in_ptr3_copy = in_ptr3.clone()
# Allocate output tensors
out_ptr0 = torch.empty(size_out_ptr, device=device, dtype=torch.float32)
out_ptr1 = torch.empty(size_out_ptr, device=device, dtype=torch.float32)

# Launch the kernel
test[grid](
    in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, out_ptr1,
    xnumel,
    rnumel,
    XBLOCK=XBLOCK,
    RBLOCK=RBLOCK
)

# Optional: Verify outputs
# Transfer outputs to CPU
out0_cpu = out_ptr0.cpu()
out1_cpu = out_ptr1.cpu()

# Check for NaNs or infinities
if torch.isnan(out0_cpu).any() or torch.isinf(out0_cpu).any():
    print("Warning: Output contains NaNs or Infinities in out_ptr0.")
else:
    print("out_ptr0 looks good.")

if torch.isnan(out1_cpu).any() or torch.isinf(out1_cpu).any():
    print("Warning: Output contains NaNs or Infinities in out_ptr1.")
else:
    print("out_ptr1 looks good.")

# Optionally, print a small portion of the outputs
print("Sample of out_ptr0:", out0_cpu[:10])
print("Sample of out_ptr1:", out1_cpu[:10])

test[grid](
    in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, out_ptr1,
    xnumel,
    rnumel,
    XBLOCK=XBLOCK,
    RBLOCK=RBLOCK
)

def test_pytorch(in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel: int, rnumel: int, XBLOCK: int, RBLOCK: int) -> (torch.Tensor, torch.Tensor):
    xindex = torch.arange(0, xnumel, device=in_ptr0.device).unsqueeze(1)  # Shape: (xnumel, 1)
    rindex = torch.arange(0, rnumel, device=in_ptr0.device).unsqueeze(0)  # Shape: (1, rnumel)

    x0 = xindex % 1920  # Shape: (xnumel, 1)
    x1 = xindex // 1920  # Shape: (xnumel, 1)

    indices = x0 + (1920 * rindex) + (491520 * x1)  # Shape: (xnumel, rnumel)
    indices = indices.long().squeeze()  # Ensure indices are of type LongTensor

    tmp0 = in_ptr0[indices]  # Shape: (xnumel, rnumel)
    tmp3 = in_ptr1[indices]  # Shape: (xnumel, rnumel)
    tmp9 = in_ptr2[indices]  # Shape: (xnumel, rnumel)

    tmp11 = in_ptr3[x0.squeeze().long()]  # Shape: (xnumel,)
    tmp11_expanded = tmp11.unsqueeze(1)  # Shape: (xnumel, 1)

    tmp0 = tmp0.float()
    tmp3 = tmp3.float()
    tmp9 = tmp9.float()

    tmp2 = tmp0 <= 0.0  # Shape: (xnumel, rnumel)
    tmp4 = torch.where(tmp2, torch.zeros_like(tmp0), tmp3)  # Shape: (xnumel, rnumel)
    tmp5 = tmp4.float()  # Shape: (xnumel, rnumel)

    tmp12 = tmp9 - tmp11_expanded  # Shape: (xnumel, rnumel)
    tmp13 = tmp5 * tmp12  # Shape: (xnumel, rnumel)

    tmp7 = tmp5.sum(dim=1, keepdim=True)  # Shape: (xnumel, 1)
    tmp15 = tmp13.sum(dim=1, keepdim=True)  # Shape: (xnumel, 1)

    return tmp7.squeeze(), tmp15.squeeze()

out1, out2 = test_pytorch(in_ptr0_copy, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK, RBLOCK)