import torch
import triton
import triton.language as tl


def rz_linear_forward_tl(input: torch.tensor, hashed_weight: torch.tensor,
                         M: int, K: int, N: int, H: int,
                         R3: int, R2: int, R1: int, R0: int,
                         allow_tf32: bool = True, allow_autotune: bool = True,
                         BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                         GROUP_SIZE: int = 4) -> torch.tensor:
    '''
      Compute input_tensor x hashed_weight and return an output tensor

      Args:
        input (Tensor): A MxK tensor
        hashed_weight (Tensor): A 1xH tensor
        M, K, N, H (int): Matrix dimensions
        R3, R2, R1, R0 (int): Random numbers
        allow_tf32 (bool): If tensor core is allowed
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE: Matrix tiling parameters for performance tunning

      Returns:
        output (Tensor): A MxN tensor
    '''
    # TODO(Keren): make rzlinear more general for any shape
    assert (H > (BLOCK_SIZE_K * BLOCK_SIZE_N))
    assert (M % 4 == 0)
    assert (K % 4 == 0)
    assert (N % 4 == 0)

    # allocates output
    output = torch.zeros((M, N), device=input.device, dtype=input.dtype)

    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    if allow_autotune:
        if allow_tf32:
            assert (K % 32 == 0)
            rz_linear_forward_kernel_tf32[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE
            )
        else:
            assert (K % 8 == 0)
            # XXX(Keren): triton bug, cannot materialize allow_tf32
            rz_linear_forward_kernel_fp32[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE
            )
    else:
        rz_linear_forward_kernel_notune[grid](
            input, hashed_weight, output,
            M, N, K, H,
            input.stride(0), input.stride(1),
            output.stride(0), output.stride(1),
            allow_tf32=allow_tf32,
            R3=R3, R2=R2, R1=R1, R0=R0,
            num_stages=4,
            num_warps=4,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE=GROUP_SIZE
        )
    return output


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_forward_kernel_fp32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)

    # Compute hash
    # [H]
    b_offset = b_ptr + offs_k[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
                         R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
    b_ptrs = tl.multiple_of(b_ptrs, 16)

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K//BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add K checks
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        c += tl.dot(a, b, allow_tf32=False)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs = b_offset + ((k + 1) * R3 + pid_n * R2 +
                             R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
        b_ptrs = tl.multiple_of(b_ptrs, 16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16,
                      'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_forward_kernel_tf32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)

    # Compute hash
    # [H]
    b_offset = b_ptr + offs_k[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
                         R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
    b_ptrs = tl.multiple_of(b_ptrs, 16)

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K//BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add K checks
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        c += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs = b_offset + ((k + 1) * R3 + pid_n * R2 +
                             R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
        b_ptrs = tl.multiple_of(b_ptrs, 16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def rz_linear_forward_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_cm, stride_cn,
    allow_tf32: tl.constexpr,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)

    # Compute hash
    # [H]
    b_offset = b_ptr + offs_k[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
                         R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
    b_ptrs = tl.multiple_of(b_ptrs, 16)

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K//BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add K checks
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs = b_offset + ((k + 1) * R3 + pid_n * R2 +
                             R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N) // 4 * 4
        b_ptrs = tl.multiple_of(b_ptrs, 16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
