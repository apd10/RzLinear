import torch
from torch.nn.parameter import Parameter

from .RzLinearFunction import RzLinearFunction


class RzLinear(torch.nn.Module):
    # TODO(Keren): triton int64 overflow bug?
    #P = 2038074743
    P = 56598313
    R = 4

    '''
        Args:
            P (int): The prime number used in the hash function
            R (int): Number of random numbers
    '''

    def __init__(self, input_dim: int, output_dim: int, compress_ratio: float = 0.0625, chunk_size: int = 1,
                 hashed_weight: torch.tensor = None, seed: int = 1024, bias: bool = False,
                 dtype: torch.dtype = torch.float32) -> None:
        '''
            A Linear layer using ROBE-Z compression

            Args:
                input_dim (int): Number of features in each input sample 
                output_dim (int): Number of features in each output sample
                compress_ratio (float): The compress ratio of the hashed_weight comparing to (input_dim, output_dim)
                chunk_size (int): The size of the minimal hash unit. It is unused for now
                hashed_weight (Tensor): If hashed_weight is not None, we ignore hash_size and reuse hashed_weight.
                seed (int): The random seed to init random numbers
                bias (bool): If True, adds a learnable bias to the output
                dtype (float): The default data type of parameters
        '''
        super(RzLinear, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._compress_ratio = compress_ratio
        self._chunk_size = chunk_size
        self._hashed_weight = hashed_weight
        self._bias = bias

        # random numbers are always on the CPU
        self._random_numbers = self._generate_random_numbers(seed)

        # weight
        if hashed_weight is None:
            self._hashed_weight = Parameter(
                torch.arange(int(input_dim * output_dim * compress_ratio)).type(dtype))
        else:
            self._hashed_weight = Parameter(hashed_weight)

        # bias term
        if bias:
            self._bias = Parameter(torch.zeros(self._output_dim, dtype=dtype))

    def _generate_random_numbers(self, seed: int):
        torch.manual_seed(seed)
        x = torch.randint(0, RzLinear.P, (RzLinear.R - 1,)).type(
            torch.int32).requires_grad_(False)
        x = x + x % 2
        x = torch.cat([torch.tensor([RzLinear.P], dtype=torch.int32), x])
        return x.requires_grad_(False).cpu()

    def forward(self, x) -> torch.Tensor:
        '''
            RzLinear forward function, which computes rzlinear and bias (if any)

            Args:
                input (Tensor): (N, *, input_dim), where N is the batch size

            Returns:
                output (Tensor): (N, output_dim)
        '''
        assert(len(x.shape) >= 2)
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        x = RzLinearFunction.apply(
            x, self._hashed_weight, self._random_numbers, self._output_dim, self._chunk_size)
        if self._bias is not None:
            x = x + self._bias
        return x
