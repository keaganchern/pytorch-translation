import triton
import triton.language as tl
import torch
from triton.language.extra.libdevice import rsqrt
from typing import Tuple




def test(
    X: torch.Tensor,  # input tensor, shape (n_rows, n_groups, hidden_size)
    W: torch.Tensor,  # weight tensor, shape (num_channels,)
    B: torch.Tensor,  # bias tensor, shape (num_channels,)
    channels_per_group: int,  # number of channels per group
    eps: float  # epsilon value for numerical stability
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows, n_groups, hidden_size = X.shape
    hidden_size_per_channel = hidden_size // channels_per_group

    # Compute mean and variance over hidden_size dimension per (batch_idx, group_idx)
    m = X.mean(dim=2)  # Shape (n_rows, n_groups)
    variance = X.var(dim=2, unbiased=False)  # Shape (n_rows, n_groups)
    rstd = 1.0 / torch.sqrt(variance + eps)  # Shape (n_rows, n_groups)

    # Reshape m and rstd for broadcasting
    m = m.unsqueeze(-1).unsqueeze(-1)  # Shape (n_rows, n_groups, 1, 1)
    rstd = rstd.unsqueeze(-1).unsqueeze(-1)  # Shape (n_rows, n_groups, 1, 1)

    # Reshape X to expose channels
    X = X.view(n_rows, n_groups, channels_per_group, hidden_size_per_channel)

    # Reshape W and B to broadcast over n_rows and hidden_size_per_channel
    num_channels = W.shape[0]

    assert num_channels == n_groups * channels_per_group, "Number of channels must equal n_groups * channels_per_group"
    W = W.view(n_groups, channels_per_group).unsqueeze(0).unsqueeze(-1)  # Shape (1, n_groups, channels_per_group, 1)
    B = B.view(n_groups, channels_per_group).unsqueeze(0).unsqueeze(-1)  # Shape (1, n_groups, channels_per_group, 1)

    # Compute Y
    Y = (X - m) * rstd * W + B  # Shape (n_rows, n_groups, channels_per_group, hidden_size_per_channel)

    # Reshape Y back to original shape
    Y = Y.view(n_rows, n_groups, hidden_size)

    # Squeeze m and rstd back to (n_rows, n_groups)
    m = m.squeeze(-1).squeeze(-1)  # Shape (n_rows, n_groups)
    rstd = rstd.squeeze(-1).squeeze(-1)  # Shape (n_rows, n_groups)

    return Y, m, rstd


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride,  # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    W_ptr,  # pointer to W
    B_ptr,  # pointer to B
    hidden_size,  # hidden size of X
    channels_per_group,  # the number of channels per group
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

    block_range = tl.arange(0, BLOCK_SIZE)

    # Compute mean and variance using the online algorithm
    s = 0.0
    squared_sum = 0.0
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        s += tl.sum(X)
        # X**2
        squared_sum += tl.sum(X * X)

    m = s / hidden_size

    # variance = E[X**2] - E[X]**2
    variance = (squared_sum / hidden_size) - (m * m)

    # 1/std
    rstd = rsqrt(variance + eps)

    # Normalize
    hidden_size_per_channel = hidden_size // channels_per_group
    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        W = tl.load(W_ptr + channel_idx)
        B = tl.load(B_ptr + channel_idx)
        for i in range(0, hidden_size_per_channel, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size_per_channel
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
            Y = (X - m) * rstd * W + B
            tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

        X_ptr += hidden_size_per_channel
        Y_ptr += hidden_size_per_channel

    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    # Reshape X so that the mean and std are computed across the groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)

    _group_norm_forward_kernel[(batch_size, num_groups)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        hidden_size,
        channels_per_group,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Return tensors in the original shape
    return Y.view(*shape), X.view(*shape), Mean, RSTD, BLOCK_SIZE

def run_test_case(batch_size, num_channels, feature_size, num_groups, eps=1e-5):
    # Create random input tensor with shape (batch_size, num_channels, feature_size)
    X = torch.randn(batch_size, num_channels, feature_size, device="cuda")
    # W and B have shape (num_channels,)
    W = torch.randn(num_channels, device="cuda")
    B = torch.randn(num_channels, device="cuda")
    # Execute the test case
    outputs = group_norm_forward(X, num_channels, num_groups, W, B, eps)
    # Return both inputs and outputs for further use
    return {"X": X, "W": W, "B": B}, {"Y": outputs[0], "X_out": outputs[1], "Mean": outputs[2], "RSTD": outputs[3], "BLOCK_SIZE": outputs[4]}


    # Small test case: batch_size=2, num_channels=4, feature_size=8, num_groups=2
inputs_small, outputs_small = run_test_case(batch_size=2, num_channels=4, feature_size=8, num_groups=2)



    # Medium test case: batch_size=4, num_channels=8, feature_size=16, num_groups=4
inputs_medium, outputs_medium = run_test_case(batch_size=4, num_channels=8, feature_size=16, num_groups=4)


    # Large test case: batch_size=8, num_channels=16, feature_size=64, num_groups=4
inputs_large, outputs_large = run_test_case(batch_size=8, num_channels=16, feature_size=64, num_groups=4)


opt_test = torch.compile(test)

def run_pytorch_test_case(inputs, triton_outputs, num_groups, eps=1e-5):
    X_orig = inputs["X"]  # shape: (batch_size, num_channels, feature_size)
    batch_size, num_channels, feature_size = X_orig.shape
    # Reshape X the same way as in the Triton test: (batch_size, num_groups, -1)
    X_reshaped = X_orig.view(batch_size, num_groups, -1)
    channels_per_group = num_channels // num_groups

    # Compute PyTorch output

    Y_pt, m_pt, rstd_pt = opt_test(X_reshaped, inputs["W"], inputs["B"], channels_per_group, eps)

    # Triton kernel returned Y in the original input shape; reshape it for comparison.
    Y_triton = triton_outputs["Y"].view(batch_size, num_groups, -1)
    Mean_triton = triton_outputs["Mean"]
    RSTD_triton = triton_outputs["RSTD"]

    assert(torch.allclose(Y_pt, Y_triton, atol=1e-4))
    assert(torch.allclose(m_pt, Mean_triton, atol=1e-4))
    assert(torch.allclose(rstd_pt, RSTD_triton, atol=1e-4))

    return {"Y": Y_pt, "Mean": m_pt, "RSTD": rstd_pt}


# Small test case: batch_size=2, num_channels=4, feature_size=8, num_groups=2

run_pytorch_test_case(inputs_small, outputs_small, num_groups=2)

# Medium test case: batch_size=4, num_channels=8, feature_size=16, num_groups=4
run_pytorch_test_case(inputs_medium, outputs_medium, num_groups=4)

# Large test case: batch_size=8, num_channels=16, feature_size=64, num_groups=4

run_pytorch_test_case(inputs_large, outputs_large, num_groups=4)




