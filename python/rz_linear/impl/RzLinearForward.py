import torch
import triton
import triton.language as tl


def rz_linear_forward_tl(input: torch.tensor, hashed_weight: torch.tensor,
                         M: int, K: int, N: int, H: int,
                         R7: int, R6: int, R5: int, R4: int,
                         R3: int, R2: int, R1: int, R0: int,
                         allow_tf32: bool = True, allow_autotune: bool = True,
                         BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                         GROUP_SIZE: int = 4, is_hnet: bool = False) -> torch.tensor:
    '''
      Compute input_tensor x hashed_weight and return an output tensor

      Args:
        input (Tensor): A MxK tensor
        hashed_weight (Tensor): A 1xH tensor
        M, K, N, H (int): Matrix dimensions
        R3, R2, R1, R0 (int): Random numbers
        allow_tf32 (bool): If tensor core is allowed
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE: Matrix tiling parameters for performance tunning
        is_hnet (bool): Baseline hnet

      Returns:
        output (Tensor): A MxN tensor
    '''
    assert (H > (BLOCK_SIZE_K * BLOCK_SIZE_N))

    # allocates output
    output = torch.zeros((M, N), device=input.device, dtype=input.dtype)

    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    if allow_autotune:
        if allow_tf32:
            rz_linear_forward_kernel_tf32[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_K=(K % BLOCK_SIZE_K == 0)
            )
        else:
            rz_linear_forward_kernel_fp32[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_K=(K % BLOCK_SIZE_K == 0)
            )
    else:
        if not is_hnet:
            rz_linear_forward_kernel_notune[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                allow_tf32=allow_tf32,
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                num_stages=4,
                num_warps=4,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_K=(K % 32 == 0)
            )
        else:
            hnet_forward_kernel_notune[grid](
                input, hashed_weight, output,
                M, N, K, H,
                input.stride(0), input.stride(1),
                output.stride(0), output.stride(1),
                allow_tf32=allow_tf32,
                R7=R7, R6=R6, R5=R5, R4=R4,
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
    R7: int, R6: int, R5: int, R4: int,
    R3: int, R2: int, R1: int, R0: int,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_K: tl.constexpr
):
    rz_linear_forward_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                           stride_am=stride_am, stride_ak=stride_ak, stride_cm=stride_cm, stride_cn=stride_cn,
                           allow_tf32=False, R7=R7, R6=R6, R5=R5, R4=R4, R3=R3, R2=R2, R1=R1, R0=R0,
                           BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                           GROUP_SIZE=GROUP_SIZE, EVEN_K=EVEN_K)


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
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_K: tl.constexpr
):
    rz_linear_forward_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                           stride_am=stride_am, stride_ak=stride_ak, stride_cm=stride_cm, stride_cn=stride_cn,
                           allow_tf32=True, R7=R7, R6=R6, R5=R5, R4=R4,  R3=R3, R2=R2, R1=R1, R0=R0,
                           BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                           GROUP_SIZE=GROUP_SIZE, EVEN_K=EVEN_K)


@triton.jit
def hnet_forward_kernel_notune(
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
    R7: tl.constexpr, R6: tl.constexpr, R5: tl.constexpr, R4: tl.constexpr,
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    hnet_forward_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                      stride_am=stride_am, stride_ak=stride_ak, stride_cm=stride_cm, stride_cn=stride_cn,
                      allow_tf32=allow_tf32, R7=R7, R6=R6, R5=R5, R4=R4, R3=R3, R2=R2, R1=R1, R0=R0,
                      BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                      GROUP_SIZE=GROUP_SIZE)


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
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_K: tl.constexpr
):
    rz_linear_forward_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                           stride_am=stride_am, stride_ak=stride_ak, stride_cm=stride_cm, stride_cn=stride_cn,
                           allow_tf32=allow_tf32, R7=R7, R6=R6, R5=R5, R4=R4, R3=R3, R2=R2, R1=R1, R0=R0,
                           BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                           GROUP_SIZE=GROUP_SIZE, EVEN_K=EVEN_K)


@triton.jit
def rz_linear_forward_core(
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
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_K: tl.constexpr
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
    # b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
    #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    b_ptrs = b_offset + ((((0) * R3 + pid_n * R2 + R1) % R0) * R0 +
                         (((0) * R7 + pid_n * R5 + R4) % R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.)
        # We accumulate along the K dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        # TODO(aditya) temp int64 fix
        # b_ptrs = b_offset + ((k + 1) * R3 + pid_n * R2 +
        #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
        b_ptrs = b_offset + ((((k+1) * R3 + pid_n * R2 + R1) % R0) * R0 + (
            ((k+1) * R7 + pid_n * R5 + R4) % R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def hnet_forward_core(
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
    R7, R6, R5, R4,
    R3, R2, R1, R0,
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
    b_offset1 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + 0) * R1 + (
        tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R2 + R3) % R0
    b_offset2 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + 0) * R4 + (
        tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R5 + R6) % R0

    # b_ptrs = b_offset + (0 * R3 + pid_n * R2 +
    #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    b_ptrs = b_ptr + (b_offset1 * R0 + b_offset2) % H

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    #a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    #b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_zero = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    b_zero = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add K checks

        # offs_k += BLOCK_SIZE_K TODO(aditya) this throws error map::at (do not know why)
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_mask = (offs_cm[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_cn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=a_zero)
        b = tl.load(b_ptrs, mask=b_mask, other=b_zero)
        # We accumulate along the K dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        # TODO(aditya) temp int64 fix
        # b_ptrs = b_offset + ((k + 1) * R3 + pid_n * R2 +
        #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
        b_offset1 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + (k+1) * BLOCK_SIZE_K) * R1 + (
            tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R2 + R3) % R0
        b_offset2 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + (k+1) * BLOCK_SIZE_K) * R4 + (
            tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R5 + R6) % R0
        b_ptrs = b_ptr + (b_offset1 * R0 + b_offset2) % H

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
