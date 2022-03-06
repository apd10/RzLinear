import torch
from rz_linear import RzLinear
from rz_linear.impl.RzLinearIdx import rz_linear_idx_tl
from rz_linear.impl.RzLinearForward import rz_linear_forward_tl

device = torch.device('cuda:0')


def test_module():
    rz = RzLinear(output_dim=1024).to(device)
    assert(rz._output_dim == 1024)


def test_get_idx():
    def torch_get_idx(hashed_weight: torch.tensor) -> torch.tensor:
        weight = torch.empty(
            (K, N), device=hashed_weight.device, dtype=hashed_weight.dtype)
        for k in range(K // BLOCK_SIZE_K):
            for n in range(N // BLOCK_SIZE_N):
                idx = ((k * R3 + n * R2 + R1) %
                       R0) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
                hashed_weight_slice = slice(idx, idx+BLOCK_SIZE_K*BLOCK_SIZE_N)
                weight_k_slice = slice(k*BLOCK_SIZE_K, (k+1)*BLOCK_SIZE_K)
                weight_n_slice = slice(n*BLOCK_SIZE_N, (n+1)*BLOCK_SIZE_N)
                weight[weight_k_slice, weight_n_slice] = hashed_weight[hashed_weight_slice].view(
                    BLOCK_SIZE_K, BLOCK_SIZE_N)
        return weight

    M = 1024
    K = 1024
    N = 10240
    H = 10240 * 1024 // 16
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 128
    rz = RzLinear(output_dim=N, hash_size=H).to(device)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()
    weight_tl = rz_linear_idx_tl(
        rz._hashed_weight, BLOCK_SIZE_K, BLOCK_SIZE_N, K, N, H, R3, R2, R1, R0)
    weight_torch = torch_get_idx(rz._hashed_weight)
    assert(torch.allclose(weight_tl, weight_torch, rtol=1e-3) is True)


def test_forward():
    M = 256
    K = 256
    N = 256
    H = 256 * 256 // 16
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32

    input = torch.rand((M, K), device=device)
    rz = RzLinear(output_dim=N, hash_size=H).to(device)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()

    torch.backends.cuda.matmul.allow_tf32 = False
    rz_output = rz_linear_forward_tl(input, rz._hashed_weight, M, K, N, H, R3, R2,
                                     R1, R0, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N)
    weight = rz_linear_idx_tl(
        rz._hashed_weight, BLOCK_SIZE_K, BLOCK_SIZE_N, K, N, H, R3, R2, R1, R0)
    torch_output = torch.mm(input, weight)
    print(rz_output)
    print(torch_output)


def test_backward():
    pass
