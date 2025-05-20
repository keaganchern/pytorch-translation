import torch
import triton
import triton.language as tl

#source
@triton.jit
def test(
    output_ptr,  # *Pointer* to output tensor.
    source_ptr,  # *Pointer* to source tensor.
    index_ptr,  # *Pointer* to index tensor.
    num_indices,
    num_cols,
    stride0,  # Stride information of source tensor.
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices each program should process.
    BLOCK_SIZE_COL: tl.constexpr,  # Number of cols each program should process.
):
    pid0 = tl.program_id(axis=0)  # We use 2D launch grid
    pid1 = tl.program_id(axis=1)

    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=(indices < num_indices))
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)

    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)

def test_pytorch(source_tensor: torch.Tensor, index_tensor: torch.Tensor, num_indices: int, num_cols: int) -> torch.Tensor:
    """
    Selects elements from a source tensor based on an index tensor.

    Args:
        source_tensor: The tensor to select elements from.
        index_tensor: A 1D tensor containing the indices to select from the
                      0th dimension of source_tensor.
        num_indices: The number of indices to use from the beginning of index_tensor.
        num_cols: The number of columns to select from the source_tensor
                  (from the 0th column up to num_cols-1).

    Returns:
        A new tensor containing the selected elements. The shape of the output
        tensor will be (num_indices, num_cols).
    """

    
    effective_indices = index_tensor[:num_indices]
    output_tensor = source_tensor[effective_indices, :num_cols]

    return output_tensor

def index_select_cat_fwd(
    output: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
):
    if not (source.is_cuda and index.is_cuda):
        raise ValueError("The index tensor and the source tensor must be of type CUDA!")

    if not source.ndim == 2:
        raise ValueError(f"Expected 2-dimensional tensor, got {source.ndim}.")
    if not index.ndim == 1:
        raise ValueError(f"Expected 1-dimensional tensor, got {index.ndim}.")

    num_rows, num_cols = source.shape
    num_indices = index.shape[0]

    if not num_indices < num_rows:
        raise ValueError(
            "The number of indices cannot exceed the number of rows in the source matrix."
        )

    stride0, stride1 = source.stride(0), source.stride(1)

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    test[grid](
        output,
        source,
        index,
        num_indices,
        num_cols,
        stride0,
        stride1,
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=512,
    )

    return output

# The Triton kernel 'test' and the Python driver 'index_select_cat_fwd'
# are assumed to be defined in the execution environment where this script will run.
# This script only contains the testing code.

def create_and_run_triton_test_instance(source_rows: int, source_cols: int, num_indices_to_select: int):
    """
    Generates input tensors for the Triton kernel, executes it via the driver,
    and returns the input tensors and the output tensor.

    Args:
        source_rows: Number of rows for the source tensor.
        source_cols: Number of columns for the source tensor.
        num_indices_to_select: Number of indices to select.
                               Must be >= 0 and < source_rows.

    Returns:
        A tuple containing:
            - source_tensor (torch.Tensor): The generated source tensor.
            - index_tensor (torch.Tensor): The generated index tensor.
            - output_triton (torch.Tensor): The output tensor from the Triton kernel.
    Raises:
        ValueError: If input parameters violate constraints necessary for
                    the driver 'index_select_cat_fwd' to run successfully.
    """
    # Validate arguments based on driver's explicit constraints for successful execution
    # The driver 'index_select_cat_fwd' checks 'if not num_indices < num_rows:'
    # This implies num_rows must be at least 1 if num_indices is 0 (0 < num_rows).
    # And num_indices must be strictly less than num_rows.
    if not source_rows > 0:
        raise ValueError(
            f"source_rows must be > 0 to satisfy 'num_indices < num_rows' driver constraint. Got {source_rows}."
        )
    if not num_indices_to_select >= 0:
        raise ValueError(
            f"num_indices_to_select must be non-negative. Got {num_indices_to_select}."
        )
    if not num_indices_to_select < source_rows:
        raise ValueError(
            f"num_indices_to_select ({num_indices_to_select}) must be strictly less than "
            f"source_rows ({source_rows}) due to driver's 'num_indices < num_rows' constraint."
        )

    # 1. Generate random input tensors
    # Source tensor: 2D, CUDA, float32
    source_tensor = torch.randn((source_rows, source_cols), device='cuda', dtype=torch.float32)

    # Index tensor: 1D, CUDA, int64. Values must be valid row indices for source_tensor.
    if num_indices_to_select > 0:
        # Generate unique indices within the valid range [0, source_rows - 1]
        index_tensor = torch.randperm(source_rows, device='cuda')[:num_indices_to_select].to(torch.int64)
    else: # num_indices_to_select == 0
        index_tensor = torch.empty((0,), device='cuda', dtype=torch.int64)

    # 2. Prepare output tensor
    # The Triton kernel will populate this tensor.
    # Its shape should be (num_indices_to_select, source_cols).
    output_triton = torch.empty((num_indices_to_select, source_cols), device='cuda', dtype=source_tensor.dtype)

    # 3. Execute the Triton kernel using the provided driver function 'index_select_cat_fwd'
    # We assume 'index_select_cat_fwd' is available in the global scope where this script runs.
    # The driver handles its own checks (e.g., tensor dimensions, CUDA type).
    index_select_cat_fwd(
        output_triton,
        source_tensor,
        index_tensor
    )

    # 4. Return input tensors and the output tensor from Triton
    return source_tensor, index_tensor, output_triton

# Define test configurations: small, medium, and large (max 3 cases)
# Parameters are chosen to satisfy the driver's constraints:
# - source_rows > 0
# - 0 <= num_indices_to_select < source_rows
test_configs = [
    {
        "name": "small_case",
        "source_rows": 20,
        "source_cols": 128, # Less than BLOCK_SIZE_COL
        "num_indices_to_select": 5
    },
    {
        "name": "medium_case_zero_indices",
        "source_rows": 256,
        "source_cols": 512, # Equal to BLOCK_SIZE_COL
        "num_indices_to_select": 0 # Edge case: selecting zero indices
    },
    {
        "name": "large_case",
        "source_rows": 1024,
        "source_cols": 1024, # Multiple of BLOCK_SIZE_COL
        "num_indices_to_select": 128
    },
]

# List to store results (inputs and Triton output) from all test cases
all_test_case_results = []

# Execute the test cases
for config in test_configs:
    # print(f"Running Triton test case: {config['name']}") # Print statements reduced as requested
    try:
        s_tensor, idx_tensor, o_tensor_triton = create_and_run_triton_test_instance(
            source_rows=config["source_rows"],
            source_cols=config["source_cols"],
            num_indices_to_select=config["num_indices_to_select"]
        )
        all_test_case_results.append({
            "name": config["name"],
            "inputs": {
                "source": s_tensor,
                "index": idx_tensor,
            },
            "output_triton": o_tensor_triton,
            "status": "success"
        })
    except Exception as e:
        all_test_case_results.append({
            "name": config["name"],
            "inputs": config, # Store original config if generation failed early
            "output_triton": None,
            "status": "error",
            "error_message": str(e)
        })
        # print(f"Error running test case {config['name']}: {e}")


opt_test = torch.compile(test_pytorch)



for triton_result_case in all_test_case_results:
    if triton_result_case.get("status") == "success":
        case_name = triton_result_case["name"]
        
        # Retrieve inputs and Triton output from the stored results
        input_source_tensor = triton_result_case["inputs"]["source"]
        input_index_tensor = triton_result_case["inputs"]["index"]
        output_from_triton = triton_result_case["output_triton"]

        # Determine parameters for the test_pytorch function based on the input tensors
        # num_indices for test_pytorch is the number of elements in the input_index_tensor
        num_indices_to_use = input_index_tensor.shape[0]
        
        if input_source_tensor.ndim == 2:
            num_cols_to_use = input_source_tensor.shape[1]
        elif input_source_tensor.ndim == 1 and input_index_tensor.shape[0] > 0 : # Source is 1D vector, but kernel might treat it as (N,1)
            num_cols_to_use = 1 # This case might need specific handling based on kernel interpretation
                                # Assuming for 2D source as per triton kernel's `stride0, stride1` usage
        elif input_source_tensor.ndim == 0: # Scalar source
             num_cols_to_use = 1 # Or 0, depends on how kernel would handle scalar source.
                                 # The provided triton kernel implies 2D source.
        else: # Fallback for unexpected source dimensions, or if source is empty.
            num_cols_to_use = 0
            if output_from_triton.ndim > 1: # If triton output has columns, use that
                num_cols_to_use = output_from_triton.shape[1]


        output_from_pytorch = opt_test(
            input_source_tensor,
            input_index_tensor,
            num_indices_to_use,
            num_cols_to_use
        )

        assert output_from_pytorch.shape == output_from_triton.shape, \
            f"Test case '{case_name}': Shape mismatch. PyTorch: {output_from_pytorch.shape}, Triton: {output_from_triton.shape}"
        
      
        assert torch.allclose(output_from_pytorch, output_from_triton), \
            f"Test case '{case_name}': Value mismatch between PyTorch and Triton outputs."
