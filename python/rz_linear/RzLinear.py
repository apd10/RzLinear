import torch
from torch.nn.parameter import Parameter

from .RzLinearFunction import RzLinearFunction


class RzLinear(torch.nn.Module):
    P = 2038074743
    R = 4

    '''
        Args:
            P (int): The prime number used in the hash function
            R (int): Number of random numbers
    '''

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            chunk_size: int = 1,
            hashed_weight: Parameter = None,
            tiled=True,
            seed: int = 1024,
            bias: bool = True,
            dtype: torch.dtype = torch.float32,
            compress_ratio: float = 0.0625,
            is_hnet: bool = False,
            init_factor: float = 1.0,
            device: torch.device = None) -> None:
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
                device (torch.device): On which device the parameters are allocated
        '''
        super(RzLinear, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._compress_ratio = compress_ratio
        self._chunk_size = chunk_size
        self._hashed_weight = hashed_weight
        self._bias = bias
        self._seed = seed
        self._init_factor = init_factor
        self._is_hnet = is_hnet
        # random numbers are always on the CPU
        self._random_numbers = self._generate_random_numbers(seed)

        # weight
        if hashed_weight is None:
            self._hashed_weight = Parameter(torch.arange(
                int(input_dim * output_dim * compress_ratio), device=device).type(dtype))
        else:
            self._hashed_weight = hashed_weight

        # bias term
        if bias:
            self._bias = Parameter(torch.zeros(
                self._output_dim, dtype=dtype, device=device))

    def __repr__(self):
        return 'RzLinear(mm={}x{}, bias={}, seed={}, hashed_weight_size={}, hashed_weight_id={}, is_hashnet={}, init_factor={})'.format(
            self._input_dim,
            self._output_dim,
            (self._bias is not None),
            self._seed,
            self._hashed_weight.size(),
            self._hashed_weight.data_ptr(),
            self._is_hnet,
            self._init_factor)

    def _generate_random_numbers(self, seed: int):
        torch.manual_seed(seed)
        x = torch.randint(0, RzLinear.P, (RzLinear.R - 1,)).type(
            torch.int32).requires_grad_(False)
        x = torch.cat([torch.tensor([RzLinear.P], dtype=torch.int32), x])
        # print(x)
        return x.requires_grad_(False).cpu()

    def forward(self, x) -> torch.Tensor:
        '''
            RzLinear forward function, which computes rzlinear and bias (if any)
            Args:
                input (Tensor): (N, *, input_dim), where N is the batch size
            Returns:
                output (Tensor): (N, output_dim)
        '''
        assert (len(x.shape) >= 2)
        dim_gt_2 = x.dim() > 2
        if dim_gt_2:
            shape = x.shape
            x = x.reshape(-1, shape[-1]).contiguous()
        x = RzLinearFunction.apply(
            x,
            self._hashed_weight,
            self._random_numbers,
            self._output_dim,
            self._chunk_size,
            self._is_hnet,
            self._init_factor)
        if self._bias is not None:
            x = x + self._bias
        if dim_gt_2:
            x = x.view(*shape[:-1], x.shape[-1])
        return x
