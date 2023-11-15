import torch

from .impl.RzLinearBackward import rz_linear_backward_tl
from .impl.RzLinearForward import rz_linear_forward_tl

controls = {}
controls['triton_allow_tf32'] = False
controls['triton_allow_autotune'] = False


class RzLinearFunction(torch.autograd.Function):
    _memoize_dict = {}

    @staticmethod
    def forward(ctx, input: torch.tensor, hashed_weight: torch.tensor,
                random_numbers: torch.tensor, output_dim, chunk_size, is_hnet) -> torch.tensor:
        '''
            Args:
                input (Tensor): (N, input_dim), where N is the batch size
                hashed_weight (Tensor): (1xH), the compressed weight tensor
                random_numbers (Tensor): (4), (R3 * k_index + R2 * n_index  + R1) % R0
                output_dim (int): N
                chunk_size (int): The size of the minimal hash unit. It is unused for now

            Returns:
                output (Tensor): (N, output_dim)
        '''
        assert (random_numbers.numel() == 8)
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
        ), random_numbers[1].item(), random_numbers[0].item()

        R7, R6, R5, R4 = random_numbers[7].item(), random_numbers[6].item(
        ), random_numbers[5].item(), random_numbers[4].item()

        M, K, N, H = input.shape[0], input.shape[1], output_dim, hashed_weight.shape[0]
        if (M, K, N, H) in RzLinearFunction._memoize_dict and is_hnet is False:
            BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, num_warps, num_stages = RzLinearFunction._memoize_dict[(
                M, K, N, H)]
            output = rz_linear_forward_tl(input, hashed_weight, M, K, N, H, R7, R6, R5, R4, R3, R2, R1, R0,
                                          allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'], BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=num_warps, num_stages=num_stages, is_hnet=is_hnet)
        else:
            output = rz_linear_forward_tl(input, hashed_weight, M, K, N, H, R7, R6, R5, R4, R3, R2, R1, R0,
                                          allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'], is_hnet=is_hnet)
        ctx.save_for_backward(input, hashed_weight, random_numbers)
        ctx.output_dim = output_dim
        ctx.chunk_size = chunk_size
        ctx.is_hnet = is_hnet
        return output

    @staticmethod
    def backward(ctx, grad):
        input, hashed_weight, random_numbers = ctx.saved_variables
        assert (random_numbers.numel() == 8)
        output_dim = ctx.output_dim
        is_hnet = ctx.is_hnet
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
        ), random_numbers[1].item(), random_numbers[0].item()

        R7, R6, R5, R4 = random_numbers[7].item(), random_numbers[6].item(
        ), random_numbers[5].item(), random_numbers[4].item()
        M, K, N, H = input.shape[0], input.shape[1], output_dim, hashed_weight.shape[0]

        if (M, K, N, H) in RzLinearFunction._memoize_dict and is_hnet is False:
            BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, num_warps, num_stages = RzLinearFunction._memoize_dict[(
                M, K, N, H)]
            input_grad, weight_grad = rz_linear_backward_tl(input, hashed_weight, grad, M, K, N, H, R7, R6, R5, R4, R3, R2, R1,
                                                            R0, allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls[
                                                                'triton_allow_autotune'],
                                                            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                                            num_warps=num_warps, num_stages=num_stages,
                                                            is_hnet=is_hnet)
        else:
            input_grad, weight_grad = rz_linear_backward_tl(input, hashed_weight, grad, M, K, N, H, R7, R6, R5, R4, R3, R2, R1,
                                                            R0, allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'], is_hnet=is_hnet)
        return input_grad, weight_grad, None, None, None, None
