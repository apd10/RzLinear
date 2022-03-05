import torch
import triton
import triton.language as tl


def rz_linear_forward_tl(input: torch.tensor, hashed_weight: torch.tensor,
                        M: int, K: int, N: int, H: int,
                        R3: int, R2: int, R1: int, R0: int) -> torch.tensor:
    '''
      Compute input_tensor x hashed_weight and return an output tensor

      Args:
        input (Tensor): A MxK tensor
        hashed_weight (Tensor): A 1xH tensor
        M, K, N, H: matrix dimensions
        R3, R2, R1, R0: random numbers

      Returns:
        output (Tensor): A MxN tensor
    '''
    # TODO(Keren): make rzlinear more general for any shape
    BLOCK_SIZE_N=64
    BLOCK_SIZE_K=32,
    assert (H > (BLOCK_SIZE_K * BLOCK_SIZE_N))
    assert (K % 32 == 0)

    # allocates output
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # TODO(Keren): select the best configuration without expensive autotuning
    # 1D launch kernel where each block gets its own program.
    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    rz_linear_forward_kernel[grid](
        input, hashed_weight, output,
        M, N, K, H,
        R3, R2, R1, R0,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        num_warps=4,
        num_stages=3,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=4
    )
    return output


@triton.jit
def rz_linear_forward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # Random numbers
    R3, R2, R1, R0,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)

    # Compute hash
    b_offset = b_ptr + offs_k[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
                         R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs = b_offset + (k * R3 + pid_n * R2 +
                             R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
