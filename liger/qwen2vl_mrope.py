import triton
import triton.language as tl
import torch



@triton.jit
def _triton_qwen2vl_mrope(
    q_ptr,
    k_ptr,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0)

    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    t_cos = cos + pid * hd
    h_cos = t_cos + bs * sl * hd
    w_cos = h_cos + bs * sl * hd
    t_sin = sin + pid * hd
    h_sin = t_sin + bs * sl * hd
    w_sin = h_sin + bs * sl * hd

    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < hd // 2)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        # with some math, we can get:
        # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_qwen2vl_mrope[(n_row,)](
        q,
        k,
        cos,
        sin,
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


# Assume the Triton kernel _triton_qwen2vl_mrope and the driver function qwen2vl_mrope_forward
# are defined elsewhere and accessible in the execution environment.

# ---- Test Case Generation ----

def Test(batch_size: int, seq_len: int, n_q_head: int, n_kv_head: int, head_dim: int, dtype=torch.float16, device='cuda'):
    """
    Generates random inputs, runs the Triton kernel via the driver function,
    and returns the inputs and outputs.

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        n_q_head (int): Number of query heads.
        n_kv_head (int): Number of key/value heads.
        head_dim (int): Head dimension.
        dtype (torch.dtype): Data type for tensors (default: torch.float16).
        device (str): Device to run on (default: 'cuda').

    Returns:
        dict: A dictionary containing input and output tensors.
            'q_in': Input query tensor.
            'k_in': Input key tensor.
            'cos_in': Input cosine tensor.
            'sin_in': Input sine tensor.
            'mrope_section_in': Input mrope_section tuple.
            'q_out': Output query tensor from Triton.
            'k_out': Output key tensor from Triton.
    """

    # Ensure head dimension is even for RoPE calculations
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even.")

    # Define shapes - Driver expects (bs, n_head, sl, hd)
    q_shape = (batch_size, n_q_head, seq_len, head_dim)
    k_shape = (batch_size, n_kv_head, seq_len, head_dim)

    # Generate random input tensors
    # Requires_grad=False is default, but explicitly setting for clarity
    q_in = torch.randn(q_shape, dtype=dtype, device=device, requires_grad=False)
    k_in = torch.randn(k_shape, dtype=dtype, device=device, requires_grad=False)

    # Generate cos/sin tensors based on kernel indexing expectations:
    # The kernel accesses elements via pid * hd + [0, bs * sl * hd, 2 * bs * sl * hd].
    # This implies a flattened structure [T_data | H_data | W_data],
    # where each section has size bs * sl * hd.
    # Within each section, data for pid is at pid * hd.
    cos_sin_shape = (3 * batch_size * seq_len * head_dim,)
    cos_in = torch.randn(cos_sin_shape, dtype=dtype, device=device, requires_grad=False)
    sin_in = torch.randn(cos_sin_shape, dtype=dtype, device=device, requires_grad=False)

    # Define mrope_section: (t_end, h_section_len)
    # Ensure 0 <= t_end <= t_end + h_section_len <= head_dim // 2
    t_end = head_dim // 8
    h_section_len = head_dim // 8 # Length of the h section
    mrope_section_in = (t_end, h_section_len)
    if not (0 <= mrope_section_in[0] <= mrope_section_in[0] + mrope_section_in[1] <= head_dim // 2):
         print(f"Warning: Adjusting mrope_section {mrope_section_in} for head_dim {head_dim}")
         t_end = head_dim // 4
         h_section_len = head_dim // 4
         mrope_section_in = (t_end, h_section_len)
         if not (0 <= mrope_section_in[0] <= mrope_section_in[0] + mrope_section_in[1] <= head_dim // 2):
             # Fallback if still invalid
             t_end = head_dim // 2
             h_section_len = 0
             mrope_section_in = (t_end, h_section_len)
             print(f"Using fallback mrope_section: {mrope_section_in}")


    # Clone inputs before passing to the function, as it might modify them inplace
    q_cloned = q_in.clone().detach()
    k_cloned = k_in.clone().detach()
    cos_cloned = cos_in.clone().detach()
    sin_cloned = sin_in.clone().detach()

    # Execute the Triton kernel via the driver function
    # The driver handles transpositions internally
    q_out, k_out, _, _ = qwen2vl_mrope_forward(
        q_cloned,
        k_cloned,
        cos_cloned,
        sin_cloned,
        mrope_section_in
    )

    # Store results
    results = {
        'q_in': q_in,
        'k_in': k_in,
        'cos_in': cos_in,
        'sin_in': sin_in,
        'mrope_section_in': mrope_section_in,
        'q_out': q_out,
        'k_out': k_out
    }
    return results

# ---- Test Execution ----

if __name__ == "__main__":
    # Ensure CUDA is available


        # Define test configurations
        test_configs = [
            # Small
            {"batch_size": 2, "seq_len": 64, "n_q_head": 8, "n_kv_head": 8, "head_dim": 64},
            # Medium (GQA example)
            {"batch_size": 4, "seq_len": 128, "n_q_head": 16, "n_kv_head": 4, "head_dim": 128},
            # Large
            {"batch_size": 1, "seq_len": 512, "n_q_head": 32, "n_kv_head": 32, "head_dim": 256},
        ]

        all_results = []

        # Run tests
        for i, config in enumerate(test_configs):
            try:

                result = Test(**config)
                all_results.append(result)


            except Exception as e:

                import traceback
                traceback.print_exc()


# function signature
def test_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sl: int,
    bs: int,
    n_qh: int,
    n_kh: int,
    hd: int,
    mrope_section_t: int,
    mrope_section_h: int,
    BACKWARD_PASS: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch equivalent of the RoPE computation part of the Triton kernel.

    Args:
        q: Query tensor with shape (bs, sl, n_qh, hd) or compatible.
        k: Key tensor with shape (bs, sl, n_kh, hd) or compatible.
        cos: Cosine values tensor with shape (bs, sl, 3, hd // 2).
             Dim 2 corresponds to t, h, w sections.
        sin: Sine values tensor with shape (bs, sl, 3, hd // 2).
             Dim 2 corresponds to t, h, w sections.
        sl: Sequence length (used for potential shape checks, consistent with Triton).
        bs: Batch size (used for potential shape checks, consistent with Triton).
        n_qh: Number of query heads (used for potential shape checks, consistent with Triton).
        n_kh: Number of key heads (used for potential shape checks, consistent with Triton).
        hd: Head dimension (must be even).
        mrope_section_t: End index for the 't' section in the head dimension (exclusive).
        mrope_section_h: Length of the 'h' section in the head dimension.
        BACKWARD_PASS: Flag to determine forward or backward RoPE application.

    Returns:
        A tuple containing the modified query and key tensors (new_q, new_k).
    """

    # Assume input tensor shapes are (bs, sl, n_heads, hd)
    # Derive hd // 2 for convenience
    hd_half = hd // 2

    # Determine RoPE boundaries
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    # Prepare masks for t, h, w sections based on head dimension indices
    # Shape: (hd // 2)
    cos_offsets = torch.arange(0, hd_half, device=q.device)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    # The Triton code implies w_mask covers up to hd // 2, consistent with arange upper bound
    w_mask = (h_end <= cos_offsets) # & (cos_offsets < hd_half) is implicit

    # Extract t, h, w components from input cos/sin tensors
    # Shape: (bs, sl, hd // 2)
    t_cos_comp = cos[:, :, 0, :]
    h_cos_comp = cos[:, :, 1, :]
    w_cos_comp = cos[:, :, 2, :]
    t_sin_comp = sin[:, :, 0, :]
    h_sin_comp = sin[:, :, 1, :]
    w_sin_comp = sin[:, :, 2, :]

    # Apply masks and sum to get the final cos_row and sin_row for each batch/seq position
    # Masks broadcast from (hd // 2) to (bs, sl, hd // 2)
    # Shape: (bs, sl, hd // 2)
    cos_row = (
        t_cos_comp * t_mask +
        h_cos_comp * h_mask +
        w_cos_comp * w_mask
    )
    sin_row = (
        t_sin_comp * t_mask +
        h_sin_comp * h_mask +
        w_sin_comp * w_mask
    )

    # Ensure cos_row and sin_row have the same dtype as q/k for computation
    cos_row = cos_row.to(q.dtype)
    sin_row = sin_row.to(q.dtype)

    # Unsqueeze cos_row and sin_row to allow broadcasting over the head dimension (n_qh or n_kh)
    # Shape: (bs, sl, 1, hd // 2)
    cos_row_b = cos_row.unsqueeze(2)
    sin_row_b = sin_row.unsqueeze(2)

    # Split query and key tensors into two halves along the head dimension
    # q_tile_1/2 shape: (bs, sl, n_qh, hd // 2)
    # k_tile_1/2 shape: (bs, sl, n_kh, hd // 2)
    q_tile_1 = q[..., :hd_half]
    q_tile_2 = q[..., hd_half:]
    k_tile_1 = k[..., :hd_half]
    k_tile_2 = k[..., hd_half:]

    # Apply Rotary Position Embedding (RoPE)
    if not BACKWARD_PASS:
        # Forward pass: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # new_q_tile_1 = q_tile_1 * cos - q_tile_2 * sin
        # new_q_tile_2 = q_tile_2 * cos + q_tile_1 * sin
        new_q_tile_1 = q_tile_1 * cos_row_b - q_tile_2 * sin_row_b
        new_q_tile_2 = q_tile_2 * cos_row_b + q_tile_1 * sin_row_b

        # new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin
        # new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin
        new_k_tile_1 = k_tile_1 * cos_row_b - k_tile_2 * sin_row_b
        new_k_tile_2 = k_tile_2 * cos_row_b + k_tile_1 * sin_row_b
    else:
        # Backward pass: dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        # This simplifies to applying the inverse rotation (using -sin)
        # new_q_tile_1 = q_tile_1 * cos + q_tile_2 * sin (equivalent to dx1*cos - (-dx2)*(-sin))
        # new_q_tile_2 = q_tile_2 * cos - q_tile_1 * sin (equivalent to dx2*cos + (-dx1)*(-sin))
        new_q_tile_1 = q_tile_1 * cos_row_b + q_tile_2 * sin_row_b
        new_q_tile_2 = q_tile_2 * cos_row_b - q_tile_1 * sin_row_b

        # new_k_tile_1 = k_tile_1 * cos + k_tile_2 * sin
        # new_k_tile_2 = k_tile_2 * cos - k_tile_1 * sin
        new_k_tile_1 = k_tile_1 * cos_row_b + k_tile_2 * sin_row_b
        new_k_tile_2 = k_tile_2 * cos_row_b - k_tile_1 * sin_row_b


    # Concatenate the modified halves back together
    # Shape: (bs, sl, n_qh, hd)
    new_q = torch.cat((new_q_tile_1, new_q_tile_2), dim=-1)
    # Shape: (bs, sl, n_kh, hd)
    new_k = torch.cat((new_k_tile_1, new_k_tile_2), dim=-1)

    return new_q, new_k


opt_test = torch.compile(test_pytorch)


# ----- PyTorch Test Execution -----



passed_count = 0
failed_count = 0

# Check if all_results exists and is not empty
if 'all_results' not in globals() or not all_results:
    print("Error: 'all_results' not found or empty. Please ensure Triton tests were run and results are available.")
else:
    for i, result in enumerate(all_results):


        # Extract data from the results dictionary
        q_in_triton = result['q_in']         # Shape (bs, n_qh, sl, hd)
        k_in_triton = result['k_in']         # Shape (bs, n_kh, sl, hd)
        cos_in_flat = result['cos_in']       # Shape (3 * bs * sl * hd,)
        sin_in_flat = result['sin_in']       # Shape (3 * bs * sl * hd,)
        mrope_section_in = result['mrope_section_in'] # (t_end, h_len)
        q_out_triton = result['q_out']       # Shape (bs, n_qh, sl, hd)
        k_out_triton = result['k_out']       # Shape (bs, n_kh, sl, hd)

        # Get shape parameters and device/dtype
        bs, n_qh, sl, hd = q_in_triton.shape
        n_kh = k_in_triton.shape[1]
        device = q_in_triton.device
        dtype = q_in_triton.dtype
        hd_half = hd // 2

        mrope_section_t, mrope_section_h = mrope_section_in

        # 1. Prepare inputs for the PyTorch function

        # Transpose q and k to (bs, sl, n_heads, hd) as expected by test_pytorch
        q_pytorch_in = q_in_triton.transpose(1, 2).contiguous()
        k_pytorch_in = k_in_triton.transpose(1, 2).contiguous()

        # Reshape flat cos/sin from Triton test to (bs, sl, 3, hd // 2) for PyTorch function
        try:
            # Reshape cos
            cos_sections = cos_in_flat.view(3, bs * sl * hd)
            cos_reshaped = cos_sections.view(3, bs * sl, hd)[:, :, :hd_half] # Shape (3, bs*sl, hd//2)
            # Permute and view: (bs*sl, 3, hd//2) -> (bs, sl, 3, hd//2)
            cos_pytorch_in = cos_reshaped.permute(1, 0, 2).view(bs, sl, 3, hd_half)

            # Reshape sin
            sin_sections = sin_in_flat.view(3, bs * sl * hd)
            sin_reshaped = sin_sections.view(3, bs * sl, hd)[:, :, :hd_half] # Shape (3, bs*sl, hd//2)
            # Permute and view: (bs*sl, 3, hd//2) -> (bs, sl, 3, hd//2)
            sin_pytorch_in = sin_reshaped.permute(1, 0, 2).view(bs, sl, 3, hd_half)

        except Exception as e:
            print(f"Error reshaping cos/sin for PyTorch: {e}")
            print(f"  Flat shape: {cos_in_flat.shape}, Target shape: ({bs}, {sl}, 3, {hd_half})")
            failed_count += 1
            continue

        # 2. Run the PyTorch implementation
        try:

            q_out_pytorch_raw, k_out_pytorch_raw = opt_test(
                q=q_pytorch_in,
                k=k_pytorch_in,
                cos=cos_pytorch_in,
                sin=sin_pytorch_in,
                sl=sl,
                bs=bs,
                n_qh=n_qh,
                n_kh=n_kh,
                hd=hd,
                mrope_section_t=mrope_section_t,
                mrope_section_h=mrope_section_h,
                BACKWARD_PASS=False # Assuming forward pass based on driver name
            )
        except Exception as e:
            print(f"Error running PyTorch function: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

        # 3. Transpose PyTorch output back to match Triton output shape (bs, n_heads, sl, hd)
        q_out_pytorch = q_out_pytorch_raw.transpose(1, 2)
        k_out_pytorch = k_out_pytorch_raw.transpose(1, 2)

        # 4. Compare outputs
        # Set tolerance based on dtype
        if dtype == torch.float16:
            atol = 1e-2
            rtol = 1e-2
        elif dtype == torch.bfloat16:
             atol = 1.5e-2 # bfloat16 has lower precision
             rtol = 1.5e-2
        else: # float32
            atol = 1e-5
            rtol = 1e-5

        try:
            q_match = torch.allclose(q_out_pytorch, q_out_triton, atol=atol, rtol=rtol)
            k_match = torch.allclose(k_out_pytorch, k_out_triton, atol=atol, rtol=rtol)

            if q_match and k_match:

                passed_count += 1
            else:
                print("Outputs DO NOT MATCH")
                if not q_match:
                    print("  Mismatch in Q output")
                    # Optional: Print more debug info
                    # diff_q = torch.abs(q_out_pytorch - q_out_triton)
                    # print(f"    Max abs diff Q: {diff_q.max()}")
                    # print(f"    Max rel diff Q: {(diff_q / torch.abs(q_out_triton + 1e-6)).max()}")
                if not k_match:
                    print("  Mismatch in K output")
                    # diff_k = torch.abs(k_out_pytorch - k_out_triton)
                    # print(f"    Max abs diff K: {diff_k.max()}")
                    # print(f"    Max rel diff K: {(diff_k / torch.abs(k_out_triton + 1e-6)).max()}")
                failed_count += 1

        except Exception as e:
            print(f"Error during comparison: {e}")
            failed_count += 1