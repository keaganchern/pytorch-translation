from torch._inductor.runtime import triton_helpers
import triton
import triton.language as tl
import torch




@triton.jit
def triton_per_fused__softmax__to_copy_add_div_native_dropout_6(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 98304
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.load(in_ptr1 + load_seed_offset)
    tmp15 = r1 + (512*x0)
    tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
    tmp17 = 0.1
    tmp18 = tmp16 > tmp17
    tmp19 = tmp9 / tmp13
    tmp20 = tmp18.to(tl.float32)
    tmp21 = tmp20 * tmp19
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp18, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp19, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp24, rmask)

def test_pytorch(in_ptr0: torch.Tensor, in_ptr1: torch.Tensor, load_seed_offset: int, xnumel: int, rnumel: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    device = in_ptr0.device  # Ensure all tensors are on the same device

    # Reshape in_ptr0 to (xnumel, rnumel)
    input0 = in_ptr0.view(xnumel, rnumel).to(device)

    # tmp0 = input0 / 8.0
    tmp0 = input0 / 8.0

    # tmp7 = max(tmp0, dim=1, keepdim=True)
    tmp7 = tmp0.max(dim=1, keepdim=True)[0]

    # tmp8 = tmp0 - tmp7
    tmp8 = tmp0 - tmp7

    # tmp9 = exp(tmp8)
    tmp9 = torch.exp(tmp8)

    # tmp13 = sum(tmp9, dim=1, keepdim=True)
    tmp13 = tmp9.sum(dim=1, keepdim=True)

    # tmp19 = tmp9 / tmp13
    tmp19 = tmp9 / tmp13

    # tmp14 = in_ptr1[load_seed_offset]
    tmp14 = in_ptr1[load_seed_offset]

    # Set the seed
    seed = int(tmp14.item())
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    # Generate random numbers on the same device
    tmp16 = torch.rand(xnumel, rnumel, device=device)

    # tmp18 = tmp16 > 0.1
    tmp18 = tmp16 > 0.1

    # tmp20 = tmp18.float()
    tmp20 = tmp18.float()

    # Ensure tmp20 is on the same device as tmp19
    tmp20 = tmp20.to(device)

    # tmp21 = tmp20 * tmp19
    tmp21 = tmp20 * tmp19

    # tmp22 = 1.1111111111111112
    tmp22 = 1.1111111111111112

    # tmp23 = tmp21 * tmp22
    tmp23 = tmp21 * tmp22

    # Flatten outputs
    out_ptr3 = tmp18.flatten()
    out_ptr4 = tmp19.flatten()
    out_ptr5 = tmp23.flatten()

    return out_ptr3, out_ptr4, out_ptr5


# Constants used in the kernel
XBLOCK = 1
RBLOCK = 512

def run_test_case(test_name, xnumel, rnumel, load_seed_offset=0):
    size = RBLOCK * xnumel  # Total number of elements for in_ptr0 and outputs

    device = torch.device("cuda")

    # Allocate input tensors
    in_ptr0 = torch.randn(size, device=device, dtype=torch.float32)
    in_ptr0_copy = in_ptr0.clone()

    # For in_ptr1, which is used for seed, we need at least one element beyond load_seed_offset
    in_ptr1_size = load_seed_offset + 1
    in_ptr1 = torch.randint(0, 2**31, (in_ptr1_size,), device=device, dtype=torch.int32)
    in_ptr1_copy = in_ptr1.clone()

    # Allocate output tensors
    out_ptr3 = torch.empty(size, device=device, dtype=torch.float32)
    out_ptr4 = torch.empty(size, device=device, dtype=torch.float32)
    out_ptr5 = torch.empty(size, device=device, dtype=torch.float32)

    # Calculate grid dimensions based on xnumel
    grid = (xnumel,)

    # Launch the kernel
    triton_per_fused__softmax__to_copy_add_div_native_dropout_6[grid](
        in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5,
        load_seed_offset, xnumel, rnumel
    )

    # Print out some information for the test case
    print(f"{test_name} - out_ptr3 shape: {out_ptr3.shape}")
    print(f"{test_name} - out_ptr4 shape: {out_ptr4.shape}")
    print(f"{test_name} - out_ptr5 shape: {out_ptr5.shape}")

    # Return inputs and outputs for future use in the PyTorch program
    inputs = {
        "in_ptr0": in_ptr0,
        "in_ptr1": in_ptr1,
        "load_seed_offset": load_seed_offset,
        "xnumel": xnumel,
        "rnumel": rnumel
    }
    outputs = {
        "out_ptr3": out_ptr3,
        "out_ptr4": out_ptr4,
        "out_ptr5": out_ptr5
    }
    return inputs, outputs


    # Small test case: xnumel = 128, rnumel = 64
inputs_small, outputs_small = run_test_case("Small Test Case", xnumel=128, rnumel=64)

    # Medium test case: xnumel = 1024, rnumel = 256
inputs_medium, outputs_medium = run_test_case("Medium Test Case", xnumel=1024, rnumel=256)

    # Large test case: xnumel = 98304, rnumel = 512
inputs_large, outputs_large = run_test_case("Large Test Case", xnumel=98304, rnumel=512)


def run_pytorch_tests():
    RBLOCK = 512

    # Small Test Case
    in0_small = inputs_small["in_ptr0"].view(inputs_small["xnumel"], RBLOCK)
    pytorch_out_small = test_pytorch(
        in_ptr0=in0_small,
        in_ptr1=inputs_small["in_ptr1"],
        load_seed_offset=inputs_small["load_seed_offset"],
        xnumel=inputs_small["xnumel"],
        rnumel=RBLOCK
    )
    print("Small Test Case - pytorch out_ptr4 shape:", pytorch_out_small[1].shape)
    print("Small Test Case - Equivalent to triton out_ptr4:",
          torch.allclose(pytorch_out_small[1], outputs_small["out_ptr4"], atol=1e-4))

    # Medium Test Case
    in0_medium = inputs_medium["in_ptr0"].view(inputs_medium["xnumel"], RBLOCK)
    pytorch_out_medium = test_pytorch(
        in_ptr0=in0_medium,
        in_ptr1=inputs_medium["in_ptr1"],
        load_seed_offset=inputs_medium["load_seed_offset"],
        xnumel=inputs_medium["xnumel"],
        rnumel=RBLOCK
    )
    print("Medium Test Case - pytorch out_ptr4 shape:", pytorch_out_medium[1].shape)
    print("Medium Test Case - Equivalent to triton out_ptr4:",
          torch.allclose(pytorch_out_medium[1], outputs_medium["out_ptr4"], atol=1e-4))

    # Large Test Case
    in0_large = inputs_large["in_ptr0"].view(inputs_large["xnumel"], RBLOCK)
    pytorch_out_large = test_pytorch(
        in_ptr0=in0_large,
        in_ptr1=inputs_large["in_ptr1"],
        load_seed_offset=inputs_large["load_seed_offset"],
        xnumel=inputs_large["xnumel"],
        rnumel=RBLOCK
    )
    print("Large Test Case - pytorch out_ptr4 shape:", pytorch_out_large[1].shape)
    print("Large Test Case - Equivalent to triton out_ptr4:",
          torch.allclose(pytorch_out_large[1], outputs_large["out_ptr4"], atol=1e-4))

if __name__ == "__main__":
    run_pytorch_tests()