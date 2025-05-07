import triton
import triton.language as tl
import torch
from typing import Tuple, Optional


@triton.jit
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    label_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        if beta == 0.0:  # forward KL
            Y_prob = tl.exp(Y)
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:
            X_prob = tl.exp(X)
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            Q = tl.exp(X)
            P = tl.exp(Y)
            M = beta * P + (1 - beta) * Q
            log_M = tl.log(M)

            loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
            dX = (1 - beta) * Q * (X - log_M)

        loss = loss / n_non_ignore
        dX = dX / n_non_ignore
        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)

MAX_FUSED_SIZE = 65536
def jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    # non reduction loss
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    _jsd_kernel[(n_rows,)](
        X_ptr=_input,  # input in logspace, X = log Q
        X_stride=_input.stride(-2),
        Y_ptr=target,  # ground truth in logspace, Y = log P
        Y_stride=target.stride(-2),
        loss_ptr=loss,
        loss_stride=loss.stride(-2),
        dX_ptr=dX,
        dX_stride=dX.stride(-2),
        label_ptr=(shift_labels if has_label else torch.empty(1, device=_input.device)),  # dummy ptr if no label
        beta=beta,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
    )
    # print(loss)
    loss = torch.sum(loss)
    return loss.to(_input.dtype), dX

def test(
    X: torch.Tensor,  # input in logspace, X = log Q
    Y: torch.Tensor,  # ground truth in logspace, Y = log P
    beta: float,
    n_non_ignore: int,
    ignore_index: int,
    label: Optional[torch.Tensor] = None,
    HAS_LABEL: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_cols = X.shape

    # Create an ignore mask based on the labels
    if HAS_LABEL:
        assert label is not None
        ignore_mask = (label == ignore_index)  # Shape: (batch_size,)
    else:
        ignore_mask = torch.zeros(batch_size, dtype=torch.bool, device=X.device)
    # Expand the mask to match the shape of X and Y
    ignore_mask_expand = ignore_mask[:, None]  # Shape: (batch_size, 1)

    # Compute loss and gradient dX based on beta
    if beta == 0.0:
        # Forward KL divergence
        Y_prob = torch.exp(Y)
        loss = Y_prob * (Y - X)
        dX = -Y_prob
    elif beta == 1.0:
        # Reverse KL divergence
        X_prob = torch.exp(X)
        loss = X_prob * (X - Y)
        dX = loss + X_prob
    else:
        # General case
        Q = torch.exp(X)
        P = torch.exp(Y)
        M = beta * P + (1.0 - beta) * Q
        log_M = torch.log(M)
        loss = beta * P * Y + (1.0 - beta) * Q * X - M * log_M
        dX = (1.0 - beta) * Q * (X - log_M)

    # Normalize by the number of non-ignored samples
    loss = loss / n_non_ignore
    dX = dX / n_non_ignore

    # Set loss and gradient to zero where the label is ignored
    loss = loss.masked_fill(ignore_mask_expand, 0.0)
    dX = dX.masked_fill(ignore_mask_expand, 0.0)

    return loss, dX



# It is assumed that jsd_forward is already defined and available.

def run_test_case(test_name, BT, V, beta, ignore_index, has_label):
    # Generate random input tensors in logspace (using torch.randn to simulate log probabilities)
    _input = torch.randn(BT, V, device="cuda")
    target = torch.randn(BT, V, device="cuda")
    if has_label:
        # Create shift_labels with random integers, with a chance to set some labels to ignore_index
        shift_labels = torch.randint(low=-1, high=10, size=(BT,), device="cuda")
    else:
        shift_labels = torch.empty(1, device="cuda")

    # Execute the test case
    loss, dX = jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)


    # Return both input tensors and outputs for further use in the PyTorch program
    inputs = {"_input": _input, "target": target, "shift_labels": shift_labels, "beta": beta, "ignore_index": ignore_index, "has_label": has_label}
    outputs = {"loss": loss, "dX": dX}
    return inputs, outputs


    # Small test case: BT=2, V=8, has_label True
inputs_small, outputs_small = run_test_case("Small Test Case", BT=2, V=8, beta=0.5, ignore_index=-1, has_label=True)

    # Medium test case: BT=4, V=16, has_label False
inputs_medium, outputs_medium = run_test_case("Medium Test Case", BT=4, V=16, beta=0.5, ignore_index=-1, has_label=False)

    # Large test case: BT=8, V=64, has_label True
inputs_large, outputs_large = run_test_case("Large Test Case", BT=8, V=64, beta=0.5, ignore_index=-1, has_label=True)


opt_test = torch.compile(test)


def run_pytorch_test_case(inputs, triton_outputs):
    _input = inputs["_input"]
    target = inputs["target"]
    shift_labels = inputs["shift_labels"]
    beta = inputs["beta"]
    ignore_index = inputs["ignore_index"]
    has_label = inputs["has_label"]
    BT, V = _input.shape
    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
        label = shift_labels
    else:
        n_non_ignore = BT
        label = None

    loss_pt, dX_pt = opt_test(_input, target, beta, n_non_ignore, ignore_index, label, has_label)
    loss_pt_sum = torch.sum(loss_pt)
    assert(torch.allclose(loss_pt_sum, triton_outputs["loss"], atol=1e-4))
    assert(torch.allclose(dX_pt, triton_outputs["dX"], atol=1e-4))

    return {"loss": loss_pt_sum, "dX": dX_pt}


run_pytorch_test_case("Small Test Case", inputs_small, outputs_small)
run_pytorch_test_case("Medium Test Case", inputs_medium, outputs_medium)
run_pytorch_test_case("Large Test Case", inputs_large, outputs_large)

