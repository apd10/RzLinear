import torch
import triton
import triton.language as tl


def rz_linear_idx_tl(hashed_weight: torch.tensor,
                     BLOCK_SIZE_K: int, BLOCK_SIZE_N: int,
                     K: int, N: int, H: int,
                     R3: int, R2: int, R1: int, R0: int, num_warps: int = 4) -> torch.tensor:
    '''
      Reconstruct the original weight tensor using the hashed weight

      Args:
        hashed_weight (Tensor): (1xH) The compressed weight tensor
        BLOCK_SIZE_K, BLOCK_SIZE_N
        M, K, N, H (int): matrix dimensions
        R3, R2, R1, R0 (int): random numbers

      Returns:
        output (Tensor): A KxN tensor
    '''
    # TODO(Keren): make rzlinear more general for any shape
    assert (H > (BLOCK_SIZE_K * BLOCK_SIZE_N))
    assert (N % BLOCK_SIZE_N == 0)
    assert (K % BLOCK_SIZE_K == 0)

    # allocates output
    weight = torch.empty((N, K), device=hashed_weight.device,
                         dtype=hashed_weight.dtype)

    def grid(META): return (
        triton.cdiv(N, META['BLOCK_SIZE_N']) *
        triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    rz_linear_idx_kernel[grid](
        hashed_weight, weight,
        H,
        R3, R2, R1, R0,
        weight.stride(0), weight.stride(1),
        num_warps=num_warps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return weight


@triton.jit
def rz_linear_idx_kernel(
    bh_ptr, b_ptr,
    # Matrix dimensions
    H,
    # Random numbers
    R3, R2, R1, R0,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_bk, stride_bn,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_k = tl.cdiv(pid, BLOCK_SIZE_N)
    pid_n = pid % BLOCK_SIZE_N

    # Compute hash
    bh_offset = bh_ptr + \
        tl.arange(0, BLOCK_SIZE_K)[
            :, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    bh_ptrs = bh_offset + ((pid_k * R3 + pid_n * R2 +
                           R1) % R0) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + tl.arange(
        0, BLOCK_SIZE_N)[None, :] * stride_bn

    bh = tl.load(bh_ptrs)
    tl.store(b_ptrs, bh)
