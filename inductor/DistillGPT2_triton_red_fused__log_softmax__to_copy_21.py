import torch

def test_pytorch_claude(input_tensor, xnumel, rnumel, XBLOCK: int, RBLOCK: int) -> torch.Tensor:
    # Original indexing suggests:
    # Block size = 50257
    # Super block size = 25731584 (50257 * 512)

    # Create output tensor of the same size as input
    output = torch.empty_like(input_tensor)

    # For each x0 in range(xnumel):
    for x0 in range(xnumel):
        # Calculate start index for this batch
        start_idx = (50257 * (x0 % 511)) + (25731584 * (x0 // 511))

        # Extract the relevant slice of data
        data_slice = input_tensor[start_idx:start_idx + rnumel]

        # Compute log softmax on this slice
        # Step 1: Find max value
        max_val = torch.max(data_slice)

        # Step 2: Compute exp(x - max)
        exp_vals = torch.exp(data_slice - max_val)

        # Step 3: Compute sum of exponentials
        sum_exp = torch.sum(exp_vals)

        # Step 4: Compute final log softmax
        log_softmax = (data_slice - max_val) - torch.log(sum_exp)

        # Store result
        output[start_idx:start_idx + rnumel] = log_softmax

    return output

xnumel = 8  # Number of elements in x-dimension
rnumel = 4  # Number of elements in r-dimension
XBLOCK = 2  # Arbitrary block size (not directly used in the function)
RBLOCK = 2
input_size = 25731584 * ((xnumel // 511) + 1) + 50257 * (xnumel % 511) + rnumel
in_tensor = torch.arange(input_size, dtype=torch.float32, device='cpu')

result_unopt = test_pytorch_claude(in_tensor, xnumel, rnumel, XBLOCK, RBLOCK)