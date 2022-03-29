import torch

import triton
import triton.language as tl
import rz_linear
from rz_linear import RzLinear
from rz_linear.impl.RzLinearBackward import rz_linear_backward_input_grad_tl, rz_linear_backward_weight_grad_tl
from rz_linear.RzLinearFunction import controls

configs = [(32, 1024, 1024), (32, 10240, 10240), (128, 1024, 1024),
           (128, 10240, 10240), (1024, 1024, 1024), (1024, 10240, 10240)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        # different possible values for `x_name`
        x_vals=list(range(len(configs))),
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['cublas', 'rzlinear'],
        # label name for the lines
        line_names=['cuBLAS', 'RZLinear'],
        # line styles
        styles=[('green', '-'), ('red', 'dashed')],
        ylabel="TFLOPS",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="forward-performance",
        args={},
    )
)
def benchmark_forward(M, N, K, provider):
    # XXX(Keren): workaround, triton does not support tuple values for now
    M, K, N = configs[M]
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b))
    if provider == 'rzlinear':
        rz = RzLinear(input_dim=K, output_dim=N).to('cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rz(a))

    def perf(ms): return (2 * M * N * K * 1e-12) / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        # different possible values for `x_name`
        x_vals=list(range(len(configs))),
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['cublas', 'rzlinear'],
        # label name for the lines
        line_names=['cuBLAS', 'RZLinear'],
        # line styles
        styles=[('green', '-'), ('red', 'dashed')],
        ylabel="TFLOPS",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="backward-weight-performance",
        args={},
    )
)
def benchmark_backward_weight(M, N, K, provider):
    # XXX(Keren): workaround, triton does not support tuple values for now
    M, K, N = configs[M]
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    c = torch.randn((M, N), device='cuda', dtype=torch.float32)

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a.t(), c))
    if provider == 'rzlinear':
        rz = RzLinear(input_dim=K, output_dim=N).to('cuda')
        H = int(K * N * rz._compress_ratio)
        R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
        ), rz._random_numbers[1].item(), rz._random_numbers[0].item()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rz_linear_backward_weight_grad_tl(
                a, c, M, K, N, H, R3, R2, R1, R0, allow_tf32=controls['triton_allow_tf32']))

    def perf(ms): return (2 * M * N * K * 1e-12) / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        # different possible values for `x_name`
        x_vals=list(range(len(configs))),
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['cublas', 'rzlinear'],
        # label name for the lines
        line_names=['cuBLAS', 'RZLinear'],
        # line styles
        styles=[('green', '-'), ('red', 'dashed')],
        ylabel="TFLOPS",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="backward-input-performance",
        args={},
    )
)
def benchmark_backward_input(M, N, K, provider):
    # XXX(Keren): workaround, triton does not support tuple values for now
    M, K, N = configs[M]
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.randn((M, N), device='cuda', dtype=torch.float32)

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(c, b.t()))
    if provider == 'rzlinear':
        rz = RzLinear(input_dim=K, output_dim=N).to('cuda')
        H = int(K * N * rz._compress_ratio)
        R3, R2, R1, R0 = rz._random_numbers[3].item(), rz._random_numbers[2].item(
        ), rz._random_numbers[1].item(), rz._random_numbers[0].item()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rz_linear_backward_input_grad_tl(
                c, rz._hashed_weight, M, K, N, H, R3, R2, R1, R0,
                allow_tf32=controls['triton_allow_tf32']))

    def perf(ms): return (2 * M * N * K * 1e-12) / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


print('TF32')
controls['triton_allow_autotune'] = True
benchmark_forward.run(show_plots=True, print_data=True)
benchmark_backward_weight.run(show_plots=True, print_data=True)
benchmark_backward_input.run(show_plots=True, print_data=True)

print('Float32')
torch.backends.cuda.matmul.allow_tf32 = False
controls['triton_allow_tf32'] = False
benchmark_forward.run(show_plots=True, print_data=True)
benchmark_backward_weight.run(show_plots=True, print_data=True)
benchmark_backward_input.run(show_plots=True, print_data=True)
