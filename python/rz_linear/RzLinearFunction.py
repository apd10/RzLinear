import torch

from .impl.RzLinearForward import rz_linear_forward_tl
from .impl.RzLinearBackward import rz_linear_backward_tl


controls = {}
controls['triton_allow_tf32'] = False
controls['triton_allow_autotune'] = False


class RzLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, hashed_weight: torch.tensor,
                random_numbers: torch.tensor, output_dim, chunk_size, init_factor) -> torch.tensor:
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
        assert(random_numbers.numel() == 8)
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
        ), random_numbers[1].item(), random_numbers[0].item()

        R7, R6, R5, R4 = random_numbers[7].item(), random_numbers[6].item(
        ), random_numbers[5].item(), random_numbers[4].item()

        M, K, N, H = input.shape[0], input.shape[1], output_dim, hashed_weight.shape[0]
        # TODO(Keren): select the best configuration without expensive autotuning
        # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE = rz_linear_forward_config_tl
        output = rz_linear_forward_tl(input.contiguous(), hashed_weight.data.contiguous(),init_factor, M, K, N, H, R7, R6, R5, R4, R3, R2, R1, R0,
                                      allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'])
        ctx.save_for_backward(input, hashed_weight, random_numbers)
        ctx.output_dim = output_dim
        ctx.chunk_size = chunk_size
        ctx.init_factor = init_factor
        return output

    @staticmethod
    def backward(ctx, grad):
        input, hashed_weight, random_numbers = ctx.saved_variables
        assert(random_numbers.numel() == 8)
        output_dim = ctx.output_dim
        chunk_size = ctx.chunk_size
        init_factor = ctx.init_factor
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
        ), random_numbers[1].item(), random_numbers[0].item()

        R7, R6, R5, R4 = random_numbers[7].item(), random_numbers[6].item(
        ), random_numbers[5].item(), random_numbers[4].item()
        M, K, N, H = input.shape[0], input.shape[1], output_dim, hashed_weight.shape[0]
        input_grad, weight_grad = rz_linear_backward_tl(input.contiguous(), hashed_weight, grad.contiguous(), init_factor, M, K, N, H, R7, R6, R5, R4, R3, R2, R1,
                                                        R0, allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'])
        return input_grad, weight_grad, None, None, None, None
