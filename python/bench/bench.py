import torch

import triton
import triton.language as tl
from rz_linear import RzLinear


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        x_vals=[1024, 10240],  # different possible values for `x_name`
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
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    # XXX(Keren): workaround, triton does not support tuple values for now
    if M == 10240:
        M = 1024
        N = 10240
        K = 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b))
    if provider == 'rzlinear':
        H = N * K // 16
        rz = RzLinear(output_dim=N, hash_size=H).to('cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rz(a))

    def perf(ms): return (2 * M * N * K * 1e-12) / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
