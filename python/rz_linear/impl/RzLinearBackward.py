from typing import Tuple
import torch
import triton
import triton.language as tl


def rz_linear_backward_tl(input: torch.tensor, hashed_weight: torch.tensor, output_grad: torch.tensor,
                          M: int, K: int, N: int, H: int,
                          R3: int, R2: int, R1: int, R0: int,
                          allow_tf32: bool = True,
                          BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                          GROUP_SIZE_M: int = 4) -> Tuple[torch.tensor, torch.tensor]:
    input_grad = rz_linear_backward_input_grad_tl(output_grad, hashed_weight, M, K, N, H, R3, R2, R1, R0, allow_tf32=allow_tf32,
                                                  BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                                  GROUP_SIZE_M=GROUP_SIZE_M)
    weight_grad = rz_linear_backward_weight_grad_tl(input, output_grad, M, K, N, H, R3, R2, R1, R0, allow_tf32=allow_tf32,
                                                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                                    GROUP_SIZE_M=GROUP_SIZE_M)
    return input_grad, weight_grad


def rz_linear_backward_weight_grad_tl(input: torch.tensor, output_grad: torch.tensor,
                                      M: int, K: int, N: int, H: int,
                                      R3: int, R2: int, R1: int, R0: int,
                                      allow_tf32: bool = True,
                                      BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                                      GROUP_SIZE_M: int = 4) -> torch.tensor:
    '''
        Compute input^T x output_grad and return a weight_grad tensor

        Args:
            input (Tensor): A MxK tensor
            output_grad (Tensor): A MxN tensor
            M, K, N, H (int): Matrix dimensions
            R3, R2, R1, R0 (int): Random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M: Matrix tiling parameters for performance tunning

        Returns:
            hashed_weight_grad (Tensor): A 1xH tensor
    '''
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    hashed_weight_grad = torch.empty(
        (K, N), device=output_grad.device, dtype=output_grad.dtype)
    # 1D launch kernel where each block gets its own program.

    def grid(META): return (
        triton.cdiv(K, META['BLOCK_SIZE_K']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    rz_linear_backward_weight_grad_kernel[grid](
        input, output_grad, hashed_weight_grad,
        M, N, K, H,
        input.stride(0), input.stride(1),
        output_grad.stride(0), output_grad.stride(1),
        R3=R3, R2=R2, R1=R1, R0=R0,
        allow_tf32=allow_tf32,
        num_warps=4,
        num_stages=3,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return hashed_weight_grad


@triton.jit
def rz_linear_backward_weight_grad_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Random numbers
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pass


def rz_linear_backward_input_grad_tl(output_grad: torch.tensor, hashed_weight: torch.tensor,
                                     M: int, K: int, N: int, H: int,
                                     R3: int, R2: int, R1: int, R0: int,
                                     allow_tf32: bool = True,
                                     BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                                     GROUP_SIZE_M: int = 4) -> torch.tensor:
    '''
        Compute output_grad x hashed_weight^T and return an input_grad tensor

        Args:
            output_grad (Tensor): A MxN tensor
            hashed_weight (Tensor): A 1xH (KxN) tensor
            M, K, N, H (int): matrix dimensions
            R3, R2, R1, R0 (int): random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M: Matrix tiling parameters for performance tunning

        Returns:
            input_grad (Tensor): A MxK tensor
    '''
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    input_grad = torch.empty(
        (M, K), device=output_grad.device, dtype=output_grad.dtype)
    # 1D launch kernel where each block gets its own program.

    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    rz_linear_backward_input_grad_kernel[grid](
        output_grad, hashed_weight, input_grad,
        M, N, K, H,
        output_grad.stride(0), output_grad.stride(1),
        input_grad.stride(0), input_grad.stride(1),
        R3=R3, R2=R2, R1=R1, R0=R0,
        allow_tf32=allow_tf32,
        num_warps=4,
        num_stages=3,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return input_grad


@triton.jit
def rz_linear_backward_input_grad_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, H,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Random numbers
    R3, R2, R1, R0,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pass
