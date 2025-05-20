import triton
import triton.language as tl
import torch

@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel,
                                 XBLOCK: tl.constexpr):

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex

    tmp0 = tl.load(in_ptr0 + x0, mask=xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, mask=xmask)

# Constants from the kernel
xnumel = 663552
size_in_ptr0 = xnumel
size_out_ptr0 = xnumel

device = torch.device('cuda')

# Allocate input tensor with random values
in_ptr0 = torch.randn(size_in_ptr0, device=device, dtype=torch.float32)

# Allocate output tensor
out_ptr0 = torch.empty(size_out_ptr0, device=device, dtype=torch.float32)

# Choose a block size
XBLOCK = 1024

# Calculate grid dimensions
grid = ( (xnumel + XBLOCK - 1) // XBLOCK, )

# # Launch the kernel
# triton_poi_fused__to_copy_14[grid](
#     in_ptr0, out_ptr0, xnumel,
#     XBLOCK=XBLOCK
# )

# # Optional: Verify outputs
# input_cpu = in_ptr0.cpu()
# output_cpu = out_ptr0.cpu()

# if torch.allclose(input_cpu, output_cpu):
#     print("Success: Output matches input.")
# else:
#     print("Error: Output does not match input.")

# # Optionally, print the output
# print("Output Tensor:", output_cpu)


triton_poi_fused__to_copy_14[grid](
    in_ptr0, out_ptr0, xnumel,
    XBLOCK=XBLOCK
)

def pytorch_test(
    in_ptr0: torch.Tensor,    # Input tensor
    xnumel: int = 663552,     # Total number of elements in x-dimension
    XBLOCK: int = 1           # Block size (similar to triton's XBLOCK)
) -> torch.Tensor:
    # Limit the input tensor to xnumel elements
    input_data = in_ptr0[:xnumel]

    # Cast the input data to float32
    tmp1 = input_data.to(torch.float32)

    return tmp1

pytorch_test(in_ptr0)