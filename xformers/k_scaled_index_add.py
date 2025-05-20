import torch
import triton
import triton.language as tl
from typing import Optional
#source
@triton.jit
def test(
    input_ptr,  # *Pointer* to input tensor.
    index_ptr,  # *Pointer* to index tensor.
    source_ptr,  # *Pointer* to source tensor.
    scaling_ptr,  # *Pointer* to the scaling tensor.
    alpha,
    num_inp_indices,
    num_src_indices,
    num_rows,
    num_cols,
    stride0,  # Stride information of input and source tensor.
    stride1,
    stride2,
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices each program should process.
    BLOCK_SIZE_ROW: tl.constexpr,  # Number of rows each program should process.
    BLOCK_SIZE_COL: tl.constexpr,  # Number of cols each program should process.
    HAS_SCALING: tl.constexpr,  # Boolean indicating if the scaling factor is present.
):
    pid0 = tl.program_id(axis=0)  # We use 3D launch grid
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)

    rows = pid1 * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    cols = pid2 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # load source
    source_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    source_offsets = (
        source_ptr
        + source_indices[:, None, None] * stride0
        + rows[None, :, None] * stride1
        + cols[None, None, :] * stride2
    )
    source_mask = (
        (source_indices[:, None, None] < num_src_indices)
        & (rows[None, :, None] < num_rows)
        & (cols[None, None, :] < num_cols)
    )
    source = tl.load(source_offsets, mask=source_mask).to(tl.float32)

    # load input
    input_indices = tl.load(
        index_ptr + source_indices, mask=(source_indices < num_src_indices)
    )
    input_offsets = (
        input_ptr
        + input_indices[:, None, None] * stride0
        + rows[None, :, None] * stride1
        + cols[None, None, :] * stride2
    )
    x = tl.load(input_offsets, mask=source_mask).to(tl.float32)

    # compute scaled index add and save
    if HAS_SCALING:
        scaling = tl.load(
            scaling_ptr + cols[None, None, :] * stride2,
            mask=(cols[None, None, :] < num_cols),
        ).to(tl.float32)
        tl.store(input_offsets, x + alpha * scaling * source, mask=source_mask)
    else:
        tl.store(input_offsets, x + alpha * source, mask=source_mask)

def test_pytorch(input_tensor: torch.Tensor,
         index_tensor: torch.Tensor,
         source_tensor: torch.Tensor,
         alpha: float,
         scaling_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Gather the slices of input_tensor according to index_tensor
    x = input_tensor[index_tensor]                  # shape: [num_src_indices, num_rows, num_cols]
    src = source_tensor                             # shape: [num_src_indices, num_rows, num_cols]

    # Compute the updated values without explicit loops
    if scaling_tensor is not None:
        # scaling_tensor: shape [num_cols] → broadcast to [1, 1, num_cols]
        scale = scaling_tensor.view(1, 1, -1)
        out = x + alpha * scale * src
    else:
        out = x + alpha * src

    # Scatter the computed results back into input_tensor
    input_tensor[index_tensor] = out

    return input_tensor

def scaled_index_add_fwd(
    x: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    scaling: Optional[torch.Tensor],
    alpha: float,
):
    if not (x.is_cuda and index.is_cuda and source.is_cuda):
        raise ValueError(
            "The input tensor, the index tensor and the source tensor must be of type CUDA!"
        )

    if not (x.ndim == 3 and source.ndim == 3):
        raise ValueError(
            f"The input and source must be three-dimensional (got {x.ndim} and {source.ndim})!"
        )
    if not x.shape[1] == source.shape[1]:
        raise ValueError(
            f"The number of elements along dimension 1 of the input and source must be the same "
            f"(got {x.shape[1], } and {source.shape[1], })!"
        )
    if not x.shape[2] == source.shape[2]:
        raise ValueError(
            f"The number of elements along dimension 2 of the input and source must be the same "
            f"(got {x.shape[2], } and {source.shape[2], })!"
        )

    num_inp_indices, num_rows, num_cols = x.shape
    num_src_indices, num_rows, num_cols = source.shape
    if not num_inp_indices >= num_src_indices:
        raise ValueError(
            f"The number of elements along dimension 0 of the input must be larger than that of source "
            f"(got {num_inp_indices} and {num_src_indices})!"
        )
    if not index.shape[0] == num_src_indices:
        raise ValueError(
            f"The number of indices and source tensors must match (got {len(index)} and {len(source)})!"
        )

    stride0, stride1, stride2 = x.stride(0), x.stride(1), x.stride(2)
    if not (
        source.stride(0) == stride0
        and source.stride(1) == stride1
        and source.stride(2) == stride2
    ):
        raise ValueError(
            f"The strides of the source and input tensors must match (got {source.stride(0)} vs. {stride0}, "
            f"{source.stride(1)} vs. {stride1}, {source.stride(2)} vs. {stride2})!"
        )

    if scaling is None:
        HAS_SCALING = False
    else:
        HAS_SCALING = True
        if not scaling.is_cuda:
            raise ValueError("The scaling tensor must be of type CUDA!")
        if not (scaling.ndim == 1 and scaling.numel() == num_cols):
            raise ValueError(
                f"The scaling tensor must be a 1-dimensional tensor (got {scaling.ndim}) and its size "
                f"must be equal to the size of dimension 2 of source (got {scaling.numel()} vs. {num_cols})."
            )
        if not scaling.stride(0) == stride2:
            raise ValueError(
                f"The stride of scaling must match the stride2 of input (got {scaling.stride(0)} vs. {stride2})"
            )

    if not index.ndim == 1:
        raise ValueError(f"The index must be one-dimensional (got {index.ndim})!")

    def grid(meta):
        return (
            triton.cdiv(num_src_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_rows, meta["BLOCK_SIZE_ROW"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    test[grid](
        x,
        index,
        source,
        scaling,
        alpha,
        num_inp_indices,
        num_src_indices,
        num_rows,
        num_cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_ROW=1,
        BLOCK_SIZE_COL=512,
        HAS_SCALING=HAS_SCALING,
    )

    return



# Small test case
def Test_small():
    # shapes: [num_inp_indices, num_rows, num_cols]
    x = torch.randn((4, 8, 16), device='cuda')
    num_src = 2
    index = torch.randint(0, x.shape[0], (num_src,), device='cuda')
    source = torch.randn((num_src, x.shape[1], x.shape[2]), device='cuda')
    scaling = torch.randn((x.shape[2],), device='cuda')
    alpha = torch.rand(1).item()
    out = scaled_index_add_fwd(x.clone(), index, source, scaling, alpha)
    return out, x, index, source, scaling, alpha

# Medium test case
def Test_medium():
    x = torch.randn((16, 32, 64), device='cuda')
    num_src = 8
    index = torch.randint(0, x.shape[0], (num_src,), device='cuda')
    source = torch.randn((num_src, x.shape[1], x.shape[2]), device='cuda')
    scaling = torch.randn((x.shape[2],), device='cuda')
    alpha = torch.randn(1).item()
    out = scaled_index_add_fwd(x.clone(), index, source, scaling, alpha)
    return out, x, index, source, scaling, alpha

# Large test case
def Test_large():
    x = torch.randn((64, 128, 256), device='cuda')
    num_src = 32
    index = torch.randint(0, x.shape[0], (num_src,), device='cuda')
    source = torch.randn((num_src, x.shape[1], x.shape[2]), device='cuda')
    scaling = torch.randn((x.shape[2],), device='cuda')
    alpha = torch.randn(1).item()
    out = scaled_index_add_fwd(x.clone(), index, source, scaling, alpha)
    return out, x, index, source, scaling, alpha

# Execute all test cases
out_small, x_small, idx_small, src_small, scale_small, a_small = Test_small()
out_medium, x_medium, idx_medium, src_medium, scale_medium, a_medium = Test_medium()
out_large, x_large, idx_large, src_large, scale_large, a_large = Test_large()

