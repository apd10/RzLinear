from typing import Tuple

import torch
import triton
import triton.language as tl


def rz_linear_backward_tl(input: torch.tensor, hashed_weight: torch.tensor, output_grad: torch.tensor,
                          M: int, K: int, N: int, H: int,
                          R7: int, R6: int, R5: int, R4: int,
                          R3: int, R2: int, R1: int, R0: int,
                          allow_tf32: bool = True, allow_autotune: bool = False,
                          BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                          GROUP_SIZE: int = 4, num_warps: int = 4, num_stages: int = 4, is_hnet: bool = False) -> Tuple[torch.tensor, torch.tensor]:
    input_grad = rz_linear_backward_input_grad_tl(output_grad, hashed_weight, M, K, N, H, R7, R6, R5, R4, R3, R2, R1, R0, allow_tf32=allow_tf32, allow_autotune=allow_autotune,
                                                  BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                                  GROUP_SIZE=GROUP_SIZE, num_warps=num_warps, num_stages=num_stages, is_hnet=is_hnet)
    weight_grad = rz_linear_backward_weight_grad_tl(input, output_grad, M, K, N, H, R7, R6, R5, R4, R3, R2, R1, R0, allow_tf32=allow_tf32, allow_autotune=allow_autotune,
                                                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                                    GROUP_SIZE=GROUP_SIZE, num_warps=num_warps, num_stages=num_stages, is_hnet=is_hnet)
    return input_grad, weight_grad


def rz_linear_backward_weight_grad_tl(input: torch.tensor, output_grad: torch.tensor,
                                      M: int, K: int, N: int, H: int,
                                      R7: int, R6: int, R5: int, R4: int,
                                      R3: int, R2: int, R1: int, R0: int,
                                      allow_tf32: bool = True, allow_autotune: bool = True,
                                      BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                                      GROUP_SIZE: int = 8, num_warps: int = 4, num_stages: int = 4, is_hnet: bool = False) -> torch.tensor:
    '''
        Compute input^T x output_grad and return a weight_grad tensor

        Args:
            input (Tensor): A MxK tensor
            output_grad (Tensor): A MxN tensor
            M, K, N, H (int): Matrix dimensions
            R3, R2, R1, R0 (int): Random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE: Matrix tiling parameters for performance tunning

        Returns:
            hashed_weight_grad (Tensor): A 1xH tensor
    '''
    # allocates output
    hashed_weight_grad = torch.zeros(
        (H), device=output_grad.device, dtype=output_grad.dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META): return (
        triton.cdiv(K, META['BLOCK_SIZE_K']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    if allow_autotune and not is_hnet:
        if allow_tf32:
            rz_linear_backward_weight_grad_kernel_tf32[grid](
                input, output_grad, hashed_weight_grad,
                M, N, K, H,
                input.stride(1), input.stride(0),
                output_grad.stride(0), output_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_M=(M % BLOCK_SIZE_M == 0)
            )
        else:
            rz_linear_backward_weight_grad_kernel_fp32[grid](
                input, output_grad, hashed_weight_grad,
                M, N, K, H,
                input.stride(1), input.stride(0),
                output_grad.stride(0), output_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_M=(M % BLOCK_SIZE_M == 0)
            )
    else:
        if is_hnet:
            hnet_backward_weight_grad_kernel_notune[grid](
                input, output_grad, hashed_weight_grad,
                M, N, K, H,
                input.stride(1), input.stride(0),
                output_grad.stride(0), output_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                allow_tf32=allow_tf32,
                GROUP_SIZE=GROUP_SIZE,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N
            )
        else:
            rz_linear_backward_weight_grad_kernel_notune[grid](
                input, output_grad, hashed_weight_grad,
                M, N, K, H,
                input.stride(1), input.stride(0),
                output_grad.stride(0), output_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                allow_tf32=allow_tf32,
                GROUP_SIZE=GROUP_SIZE,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                num_stages=num_stages,
                num_warps=num_warps,
                EVEN_M=(M % BLOCK_SIZE_M == 0)
            )

    return hashed_weight_grad


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_backward_weight_grad_kernel_fp32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_M: tl.constexpr
):
    rz_linear_backward_weight_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                                        stride_am=stride_am, stride_ak=stride_ak, stride_bm=stride_bm, stride_bn=stride_bn,
                                        R7=R7, R6=R6, R5=R5, R4=R4,
                                        R3=R3, R2=R2, R1=R1, R0=R0, allow_tf32=False,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                        GROUP_SIZE=GROUP_SIZE, EVEN_M=EVEN_M)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
    ], key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_backward_weight_grad_kernel_tf32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_M: tl.constexpr
):
    rz_linear_backward_weight_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                                        stride_am=stride_am, stride_ak=stride_ak, stride_bm=stride_bm, stride_bn=stride_bn,
                                        R7=R7, R6=R6, R5=R5, R4=R4,
                                        R3=R3, R2=R2, R1=R1, R0=R0, allow_tf32=True,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                        GROUP_SIZE=GROUP_SIZE, EVEN_M=EVEN_M)


@triton.jit
def rz_linear_backward_weight_grad_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_M: tl.constexpr
):
    rz_linear_backward_weight_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                                        stride_am=stride_am, stride_ak=stride_ak, stride_bm=stride_bm, stride_bn=stride_bn,
                                        R7=R7, R6=R6, R5=R5, R4=R4,
                                        R3=R3, R2=R2, R1=R1, R0=R0, allow_tf32=allow_tf32,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                        GROUP_SIZE=GROUP_SIZE, EVEN_M=EVEN_M)


@triton.jit
def rz_linear_backward_weight_grad_core(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_M: tl.constexpr
):
    """Kernel for computing the matmul C = A^T x B.
    A has shape (M, K), B has shape (M, N) and C has shape (K, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE)
    pid_k = first_pid_k + (pid % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    # [BLOCK_SIZE_K, BLOCK_SIZE_M]
    offs_ak = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + offs_ak[:, None] * \
        stride_am + offs_am[None, :] * stride_ak

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bm = tl.arange(0, BLOCK_SIZE_M)
    b_ptrs = b_ptr + offs_bm[:, None] * \
        stride_bm + offs_bn[None, :] * stride_bn

    # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if M is not a multiple of BLOCK_SIZE_M,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add M checks
        if EVEN_M:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            offs_m = tl.arange(0, BLOCK_SIZE_M) + m * BLOCK_SIZE_M
            a = tl.load(a_ptrs, mask=offs_m[None, :] < M, other=0.)
            b = tl.load(b_ptrs, mask=offs_m[:, None] < M, other=0.)
        # We accumulate along the M dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next M block
        a_ptrs += BLOCK_SIZE_M * stride_ak
        b_ptrs += BLOCK_SIZE_M * stride_bm

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    c_offset = c_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * \
        BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    # TODO(aditya) temporary int64 fix
    # c_ptrs = c_offset + (pid_k * R3 + pid_n * R2 +
    #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    c_ptrs = c_offset + ((((pid_k) * R3 + pid_n * R2 + R1) % R0) * R0 + (
        ((pid_k) * R7 + pid_n * R5 + R4) % R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    tl.atomic_add(c_ptrs, c)


@triton.jit
def hnet_backward_weight_grad_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    hnet_backward_weight_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr, M=M, N=N, K=K, H=H,
                                   stride_am=stride_am, stride_ak=stride_ak, stride_bm=stride_bm, stride_bn=stride_bn,
                                   R7=R7, R6=R6, R5=R5, R4=R4,
                                   R3=R3, R2=R2, R1=R1, R0=R0, allow_tf32=allow_tf32,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                   GROUP_SIZE=GROUP_SIZE)


@triton.jit
def hnet_backward_weight_grad_core(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bm, stride_bn,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = A^T x B.
    A has shape (M, K), B has shape (M, N) and C has shape (K, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE)
    pid_k = first_pid_k + (pid % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    # [BLOCK_SIZE_K, BLOCK_SIZE_M]
    offs_ak = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + offs_ak[:, None] * \
        stride_am + offs_am[None, :] * stride_ak

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bm = tl.arange(0, BLOCK_SIZE_M)
    b_ptrs = b_ptr + offs_bm[:, None] * \
        stride_bm + offs_bn[None, :] * stride_bn

    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_zero = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), dtype=tl.float32)
    b_zero = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    c = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if M is not a multiple of BLOCK_SIZE_M,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add M checks

        offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        a_mask = (offs_ck[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (offs_m[:, None] < M) & (offs_cn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=a_zero)
        b = tl.load(b_ptrs, mask=b_mask, other=b_zero)
        # We accumulate along the M dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next M block
        a_ptrs += BLOCK_SIZE_M * stride_ak
        b_ptrs += BLOCK_SIZE_M * stride_bm

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    c_offset1 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + pid_k * BLOCK_SIZE_K) * R1 + (
        tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R2 + R3) % R0
    c_offset2 = ((tl.arange(0, BLOCK_SIZE_K)[:, None] + pid_k * BLOCK_SIZE_K) * R4 + (
        tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N) * R5 + R6) % R0
    c_ptrs = c_ptr + (c_offset1 * R0 + c_offset2) % H

    tl.atomic_add(c_ptrs, c)


def rz_linear_backward_input_grad_tl(output_grad: torch.tensor, hashed_weight: torch.tensor,
                                     M: int, K: int, N: int, H: int,
                                     R7: int, R6: int, R5: int, R4: int,
                                     R3: int, R2: int, R1: int, R0: int,
                                     allow_tf32: bool = True, allow_autotune: bool = True,
                                     BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                                     GROUP_SIZE: int = 4, num_warps: int = 4, num_stages: int = 4,
                                     is_hnet: bool = False) -> torch.tensor:
    '''
        Compute output_grad x hashed_weight^T and return an input_grad tensor

        Args:
            output_grad (Tensor): A MxN tensor
            hashed_weight (Tensor): A 1xH (KxN) tensor
            M, K, N, H (int): matrix dimensions
            R3, R2, R1, R0 (int): random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE: Matrix tiling parameters for performance tunning

        Returns:
            input_grad (Tensor): A MxK tensor
    '''
    # allocates output
    input_grad = torch.empty(
        (M, K), device=output_grad.device, dtype=output_grad.dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(K, META['BLOCK_SIZE_K']),
    )

    if allow_autotune and not is_hnet:
        if allow_tf32:
            rz_linear_backward_input_grad_kernel_tf32[grid](
                output_grad, hashed_weight, input_grad,
                M, N, K, H,
                output_grad.stride(0), output_grad.stride(1),
                input_grad.stride(0), input_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_N=(N % BLOCK_SIZE_N == 0)
            )
        else:
            rz_linear_backward_input_grad_kernel_fp32[grid](
                output_grad, hashed_weight, input_grad,
                M, N, K, H,
                output_grad.stride(0), output_grad.stride(1),
                input_grad.stride(0), input_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_N=(N % BLOCK_SIZE_N == 0)
            )
    else:
        if is_hnet:
            hnet_backward_input_grad_kernel_notune[grid](
                output_grad, hashed_weight, input_grad,
                M, N, K, H,
                output_grad.stride(0), output_grad.stride(1),
                input_grad.stride(0), input_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                allow_tf32=allow_tf32,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE=GROUP_SIZE
            )
        else:
            rz_linear_backward_input_grad_kernel_notune[grid](
                output_grad, hashed_weight, input_grad,
                M, N, K, H,
                output_grad.stride(0), output_grad.stride(1),
                input_grad.stride(0), input_grad.stride(1),
                R7=R7, R6=R6, R5=R5, R4=R4,
                R3=R3, R2=R2, R1=R1, R0=R0,
                allow_tf32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE=GROUP_SIZE,
                EVEN_N=(N % BLOCK_SIZE_N == 0)
            )
    return input_grad


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 16}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_backward_input_grad_kernel_fp32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_N: tl.constexpr
):
    rz_linear_backward_input_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr,
                                       M=M, N=N, K=K, H=H,
                                       stride_am=stride_am, stride_an=stride_an,
                                       stride_cm=stride_cm, stride_ck=stride_ck,
                                       R7=R7, R6=R6, R5=R5, R4=R4,
                                       R3=R3, R2=R2, R1=R1, R0=R0,
                                       allow_tf32=False,
                                       BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                       GROUP_SIZE=GROUP_SIZE, EVEN_N=EVEN_N)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64,
                      'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=4),
    ], key=['M', 'N', 'K'],
)
@triton.jit
def rz_linear_backward_input_grad_kernel_tf32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_N: tl.constexpr
):
    rz_linear_backward_input_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr,
                                       M=M, N=N, K=K, H=H,
                                       stride_am=stride_am, stride_an=stride_an,
                                       stride_cm=stride_cm, stride_ck=stride_ck,
                                       R7=R7, R6=R6, R5=R5, R4=R4,
                                       R3=R3, R2=R2, R1=R1, R0=R0,
                                       allow_tf32=True,
                                       BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                       GROUP_SIZE=GROUP_SIZE, EVEN_N=EVEN_N)


@triton.jit
def rz_linear_backward_input_grad_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_N: tl.constexpr
):
    rz_linear_backward_input_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr,
                                       M=M, N=N, K=K, H=H,
                                       stride_am=stride_am, stride_an=stride_an,
                                       stride_cm=stride_cm, stride_ck=stride_ck,
                                       R7=R7, R6=R6, R5=R5, R4=R4,
                                       R3=R3, R2=R2, R1=R1, R0=R0,
                                       allow_tf32=allow_tf32,
                                       BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                       GROUP_SIZE=GROUP_SIZE, EVEN_N=EVEN_N)


@triton.jit
def rz_linear_backward_input_grad_core(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, EVEN_N: tl.constexpr
):
    """Kernel for computing the matmul C = (A x B^T)
    A has shape (M, N), B has shape H->(K, N) and C has shape (M, K)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * \
        stride_am + offs_an[None, :] * stride_an

    # [BLOCK_SIZE_N, BLOCK_SIZE_K]
    # Compute hash
    b_offset = b_ptr + \
        tl.arange(0, BLOCK_SIZE_N)[
            :, None] + tl.arange(0, BLOCK_SIZE_K)[None, :] * BLOCK_SIZE_N

    # TODO(aditya) temporary int64 fix
    # b_ptrs = b_offset + (pid_k * R3 + 0 * R2 +
    #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    b_ptrs = b_offset + ((((pid_k) * R3 + 0 * R2 + R1) % R0) * R0 +
                         (((pid_k) * R7 + 0 * R5 + R4) % R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        if EVEN_N:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n * BLOCK_SIZE_N
            a = tl.load(a_ptrs, offs_n[None, :] < N, other=0.)
            b = tl.load(b_ptrs, offs_n[:, None] < N, other=0.)
        # We accumulate along the N dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next N block
        a_ptrs += BLOCK_SIZE_N * stride_an

        # TODO(aditya) temporary int64 fix
        # b_ptrs = b_offset + (pid_k * R3 + (n + 1) * R2 +
        #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
        b_ptrs = b_offset + ((((pid_k) * R3 + (n + 1) * R2 + R1) % R0) * R0 + (
            ((pid_k) * R7 + (n + 1) * R5 + R4) % R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def hnet_backward_input_grad_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    hnet_backward_input_grad_core(a_ptr=a_ptr, b_ptr=b_ptr, c_ptr=c_ptr,
                                  M=M, N=N, K=K, H=H,
                                  stride_am=stride_am, stride_an=stride_an,
                                  stride_cm=stride_cm, stride_ck=stride_ck,
                                  R7=R7, R6=R6, R5=R5, R4=R4,
                                  R3=R3, R2=R2, R1=R1, R0=R0,
                                  allow_tf32=allow_tf32,
                                  BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                  GROUP_SIZE=GROUP_SIZE)


@triton.jit
def hnet_backward_input_grad_core(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_cm, stride_ck,
    # Random numbers
    R7, R6, R5, R4,
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """Kernel for computing the matmul C = (A x B^T)
    A has shape (M, N), B has shape H->(K, N) and C has shape (M, K)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * \
        stride_am + offs_an[None, :] * stride_an

    # [BLOCK_SIZE_N, BLOCK_SIZE_K]
    # Compute hash
    # b_offset = b_ptr + \
    #    tl.arange(0, BLOCK_SIZE_N)[
    #        :, None] + tl.arange(0, BLOCK_SIZE_K)[None, :] * BLOCK_SIZE_N

    # TODO(aditya) temporary int64 fix
    # b_ptrs = b_offset + (pid_k * R3 + 0 * R2 +
    #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
    # b_ptrs = b_offset + ((((pid_k) * R3 + 0 * R2 + R1)%R0) * R0 + (((pid_k) * R7 + 0 * R5 + R4)%R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)

    b_offset1 = ((tl.arange(0, BLOCK_SIZE_K)[None, :] + (pid_k) * BLOCK_SIZE_K) * R1 + (
        tl.arange(0, BLOCK_SIZE_N)[:, None] + 0 * BLOCK_SIZE_N) * R2 + R3) % R0
    b_offset2 = ((tl.arange(0, BLOCK_SIZE_K)[None, :] + (pid_k) * BLOCK_SIZE_K) * R4 + (
        tl.arange(0, BLOCK_SIZE_N)[:, None] + 0 * BLOCK_SIZE_N) * R5 + R6) % R0
    b_ptrs = b_ptr + (b_offset1 * R0 + b_offset2) % H

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_zero = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b_zero = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if N is not a multiple of BLOCK_SIZE_N,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO(Keren): Add N checks

        offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        a_mask = (offs_cm[:, None] < M) & (offs_n[None, :] < N)
        b_mask = (offs_n[:, None] < N) & (offs_ck[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=a_zero)
        b = tl.load(b_ptrs, mask=b_mask, other=b_zero)
        # We accumulate along the N dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next N block
        a_ptrs += BLOCK_SIZE_N * stride_an

        # TODO(aditya) temporary int64 fix
        # b_ptrs = b_offset + (pid_k * R3 + (n + 1) * R2 +
        #                     R1) % R0 % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
        # b_ptrs = b_offset + ((((pid_k) * R3 + (n+1) * R2 + R1)%R0) * R0 + (((pid_k) * R7 + (n+1) * R5 + R4)%R0)) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
        b_offset1 = ((tl.arange(0, BLOCK_SIZE_K)[None, :] + (pid_k) * BLOCK_SIZE_K) * R1 + (
            tl.arange(0, BLOCK_SIZE_N)[:, None] + (n + 1) * BLOCK_SIZE_N) * R2 + R3) % R0
        b_offset2 = ((tl.arange(0, BLOCK_SIZE_K)[None, :] + (pid_k) * BLOCK_SIZE_K) * R4 + (
            tl.arange(0, BLOCK_SIZE_N)[:, None] + (n + 1) * BLOCK_SIZE_N) * R5 + R6) % R0
        b_ptrs = b_ptr + (b_offset1 * R0 + b_offset2) % H

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, c, mask=c_mask)
