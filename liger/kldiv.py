from typing import Literal
import triton
import triton.language as tl
import torch





from typing import Literal
MAX_FUSED_SIZE = 65536

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE: tl.constexpr = tl.constexpr(0)
_REDUCTION_MODE_SUM: tl.constexpr = tl.constexpr(1)
_REDUCTION_MODE_MEAN: tl.constexpr = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN: tl.constexpr = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)


def kldiv_forward(y: torch.Tensor, gt: torch.Tensor, eps: float, log_target: bool = False, reduction: str = "batchmean") -> torch.Tensor:

    if not log_target:
        loss = gt * (torch.log(torch.maximum(gt, torch.tensor(eps, device=gt.device))) - y)
    else:
        loss = torch.exp(gt) * (gt - y)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    elif reduction == "batchmean":
        return (torch.sum(loss, dim=1)).sum() / y.shape[0]
    
def get_num_warps(n: int):
    if n < 1024:
        return 2
    else:
        return 4

def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):  # [BT, V]
    BT, V = y_pred.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (BT,)
    reduction = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction,
    )

    # calculated according to the reduction mode same as in Pytorch. In the later versions, `mean` will be changed to the same behavior as `batchmean`
    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # https://github.com/pytorch/pytorch/blob/d7b57c4d63edb42e1deeeba9497fcb5f1f748ff2/torch/nn/functional.py#L3372
    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / BT
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor



def generate_random_tensors(shape, log_target, device="cuda"):
    # Generate random tensor for y_pred (in log-space)
    y_pred = torch.randn(shape, device=device, dtype=torch.float32)
    # Generate random tensor for y_true
    if log_target:
      y_true = torch.randn(shape, device=device, dtype=torch.float32)
    else:
      y_true = torch.rand(shape, device=device, dtype=torch.float32)
      y_true = y_true / y_true.sum(dim=-1, keepdim=True) #normalize to sum to 1, like a probability distribution

    return y_pred, y_true


def run_test(shape, log_target, reduction, eps):
    y_pred, y_true = generate_random_tensors(shape, log_target)
    output = kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps)
    return y_pred, y_true, output


# Test cases
input_tensors = {}
output_tensors = {}
eps = 1e-10

# Test case 1: Small shape
y_pred, y_true, output = run_test((4, 16), log_target=False, reduction="batchmean", eps=eps)
input_tensors["small"] = (y_pred, y_true)
output_tensors["small"] = output

# Test case 2: Medium shape
y_pred, y_true, output = run_test((32, 128), log_target=True, reduction="none", eps=eps)
input_tensors["medium"] = (y_pred, y_true)
output_tensors["medium"] = output

# Test case 3: Large shape
y_pred, y_true, output = run_test((128, 1024), log_target=False, reduction="sum", eps=eps)
input_tensors["large"] = (y_pred, y_true)
output_tensors["large"] = output

opt_kldiv = torch.compile(kldiv_forward)

y_pred_small, y_true_small = input_tensors["small"]
y_pred_medium, y_true_medium = input_tensors["medium"]
y_pred_large, y_true_large = input_tensors["large"]

# Run PyTorch implementation with the same inputs and reduction modes

output_torch_small = opt_kldiv(y_pred_small.clone().detach(), y_true_small.clone().detach(), eps, log_target=False, reduction="batchmean")
output_torch_medium = opt_kldiv(y_pred_medium.clone().detach(), y_true_medium.clone().detach(), eps, log_target=True, reduction="none")
output_torch_large = opt_kldiv(y_pred_large.clone().detach(), y_true_large.clone().detach(), eps, log_target=False, reduction="sum")


# Compare the outputs
torch.testing.assert_close(output_torch_small, output_tensors["small"], rtol=1e-02, atol=1e-03)
torch.testing.assert_close(output_torch_medium, output_tensors["medium"], rtol=1e-02, atol=1e-03)
torch.testing.assert_close(output_torch_large, output_tensors["large"], rtol=1e-02, atol=1e-03)