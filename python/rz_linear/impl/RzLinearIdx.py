import torch
import triton
import triton.language as tl

from .RzLinearHash import rz_linear_hash_tl


def rz_linear_idx_tl(hashed_weight: torch.tensor,
                     K: int, N: int, H: int,
                     R7: int, R6: int, R5: int, R4: int,
                     R3: int, R2: int, R1: int, R0: int,
                     BLOCK_SIZE_K: int = 32, BLOCK_SIZE_N: int = 32) -> torch.tensor:
    '''
      Reconstruct the original weight tensor using the hashed weight

      Args:
        hashed_weight (Tensor): (1xH) The compressed weight tensor
        M, K, N, H (int): Matrix dimensions
        R7, R6, R5, R4, R3, R2, R1, R0 (int): Random numbers
        BLOCK_SIZE_K, BLOCK_SIZE_N (int): Workload of each GPU block

      Returns:
        output (Tensor): A KxN tensor
    '''
    # TODO(Keren): make rzlinear more general for any shape
    assert (H > (BLOCK_SIZE_K * BLOCK_SIZE_N))
    assert (K % BLOCK_SIZE_K == 0)
    assert (N % BLOCK_SIZE_N == 0)

    # allocates output
    weight = torch.empty((K, N), device=hashed_weight.device,
                         dtype=hashed_weight.dtype)

    def grid(META):
        return (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    rz_linear_idx_kernel[grid](
        hashed_weight, weight,
        K, N, H,
        R7, R6, R5, R4,
        R3, R2, R1, R0,
        weight.stride(0), weight.stride(1),
        num_warps=4,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return weight


@triton.jit
def rz_linear_idx_kernel(
    bh_ptr, b_ptr,
    # Matrix dimensions
    K, N, H,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_bk, stride_bn,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = pid // grid_n
    pid_n = pid % grid_n

    # Compute hash
    bh_offset = bh_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

    bh_ptrs = bh_offset + \
        rz_linear_hash_tl(pid_k, pid_n, R7, R6, R5, R4, R3, R2,
                          R1, R0, H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    b_ptrs = b_ptr + pid_k * BLOCK_SIZE_K * stride_bk + pid_n * BLOCK_SIZE_N * stride_bn + \
        tl.arange(0, BLOCK_SIZE_K)[:, None] * \
        stride_bk + tl.arange(0, BLOCK_SIZE_N)[None, :]

    bh = tl.load(bh_ptrs)
    tl.store(b_ptrs, bh)
