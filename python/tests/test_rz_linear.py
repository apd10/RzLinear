import torch
from rz_linear import RzLinear
from rz_linear.impl.RzLinearIdx import rz_linear_idx_tl
from rz_linear.impl.RzLinearForward import rz_linear_forward_tl
from rz_linear.impl.RzLinearBackward import rz_linear_backward_weight_grad_tl, rz_linear_backward_input_grad_tl

device = torch.device('cuda:0')


def test_module():
    rz = RzLinear(input_dim=1024, output_dim=1024).to(device)
    assert(rz._input_dim == 1024)
    assert(rz._output_dim == 1024)
    input = torch.rand((1024, 1024), device=device)
    output = rz(input)
    assert(output.shape[0] == 1024)
    assert(output.shape[1] == 1024)


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
    N = 1024
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 128
    rz = RzLinear(input_dim=K, output_dim=N).to(device)
    H = int(K * N * rz._compress_ratio)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()

    weight_tl = rz_linear_idx_tl(
        rz._hashed_weight, K, N, H, R3, R2, R1, R0, BLOCK_SIZE_K, BLOCK_SIZE_N)
    weight_torch = torch_get_idx(rz._hashed_weight)

    assert(torch.allclose(weight_tl, weight_torch, rtol=1e-3) is True)


def test_forward():
    M = 1024
    K = 1024
    N = 1024
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 64

    input = torch.rand((M, K), device=device)
    rz = RzLinear(input_dim=K, output_dim=N).to(device)
    H = int(K * N * rz._compress_ratio)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()

    # Disable tf32 in testing
    rz_output = rz_linear_forward_tl(input, rz._hashed_weight, M, K, N, H, R3, R2, R1, R0,
                                     allow_tf32=False, allow_autotune=False, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE=1)
    weight = rz_linear_idx_tl(rz._hashed_weight, K, N,
                              H, R3, R2, R1, R0, BLOCK_SIZE_K, BLOCK_SIZE_N)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch_output = torch.mm(input, weight)

    assert(torch.allclose(rz_output, torch_output, rtol=1e-3) is True)


def test_backward_weight():
    def torch_get_weight(input: torch.tensor, output: torch.tensor) -> torch.tensor:
        hashed_weight = torch.zeros(
            (H), device=output.device, dtype=output.dtype)
        weight = torch.mm(input.permute(1, 0), output)
        for k in range(K // BLOCK_SIZE_K):
            for n in range(N // BLOCK_SIZE_N):
                idx = ((k * R3 + n * R2 + R1) %
                       R0) % (H - BLOCK_SIZE_K * BLOCK_SIZE_N)
                hashed_weight_slice = slice(idx, idx+BLOCK_SIZE_K*BLOCK_SIZE_N)
                weight_k_slice = slice(k*BLOCK_SIZE_K, (k+1)*BLOCK_SIZE_K)
                weight_n_slice = slice(n*BLOCK_SIZE_N, (n+1)*BLOCK_SIZE_N)
                hashed_weight[hashed_weight_slice] += weight[weight_k_slice,
                                                             weight_n_slice].ravel()
        return hashed_weight

    M = 1024
    K = 1024
    N = 1024
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 32

    input = torch.rand((M, K), device=device)
    output = torch.rand((M, N), device=device)
    rz = RzLinear(input_dim=K, output_dim=N).to(device)
    H = int(K * N * rz._compress_ratio)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch_weight = torch_get_weight(input, output)

    # Disable tf32 in testing
    rz_weight = rz_linear_backward_weight_grad_tl(
        input, output, M, K, N, H, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE=1)

    assert(torch.allclose(rz_weight, torch_weight, rtol=1e-3) is True)


def test_backward_input():
    M = 1024
    K = 1024
    N = 1024
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 32

    output = torch.rand((M, N), device=device)
    rz = RzLinear(input_dim=K, output_dim=N).to(device)
    H = int(K * N * rz._compress_ratio)
    R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
    ), rz._random_numbers[1].item(), rz._random_numbers[0].item()

    torch.backends.cuda.matmul.allow_tf32 = False
    weight = rz_linear_idx_tl(rz._hashed_weight, K, N,
                              H, R3, R2, R1, R0, BLOCK_SIZE_K, BLOCK_SIZE_N)
    # [M, N] x [K, N]^T
    torch_input = torch.mm(output, weight.t())

    # Disable tf32 in testing
    rz_input = rz_linear_backward_input_grad_tl(
        output, rz._hashed_weight, M, K, N, H, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE=1)

    assert(torch.allclose(rz_input, torch_input, rtol=1e-3) is True)
