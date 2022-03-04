import torch
import triton
import triton.language as tl


def rzlinear_forward_tl(input_tensor: torch.tensor, hashed_weight: torch.tensor,
												M: int, K: int, N: int, H: int,
												R3: int, R2: int, R1: int, R0: int) -> torch.tensor:
		'''
			Compute input_tensor x hashed_weight and return an output tensor

			Args:
				input_tensor (Tensor): A MxK tensor
				hashed_weight (Tensor): A 1xH tensor
				M, K, N, H: matrix dimensions
				R3, R2, R1, R0: random numbers

			Returns:
				output (Tensor): A MxN tensor
		'''
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    def grid(META): return (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    rzlinear_kernel[grid](
        a, b, c,
        M, N, K, H,
        R4, R3, R2, R1, R0,
        a.stride(0), a.stride(1),
        c.stride(0), c.stride(1),
        N, 1,
    )
    return c
