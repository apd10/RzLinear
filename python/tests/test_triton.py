import torch
import triton
import triton.language as tl

device = torch.device('cuda:0')


@triton.jit
def triton_tn_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_ck, stride_cn,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Kernel for computing the matmul C = A^T x B.
    A has shape (M, K), B has shape (M, N) and C has shape (K, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = pid // num_pid_n
    pid_n = pid % num_pid_n

    # [BLOCK_SIZE_K, BLOCK_SIZE_M]
    offs_ak = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + offs_ak[:, None] * \
        stride_am + offs_am[None, :] * stride_ak

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_bm = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + offs_bm[:, None] * \
        stride_bm + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_K, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    c = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, M // BLOCK_SIZE_M):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if M is not a multiple of BLOCK_SIZE_M,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the M dimension
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next M block
        a_ptrs += BLOCK_SIZE_M * stride_ak
        b_ptrs += BLOCK_SIZE_M * stride_bm

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_ck * \
        offs_ck[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def test_triton_tn():
    M = 1024
    K = 1024
    N = 1024
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 32

    a = torch.rand((M, K), device=device)
    b = torch.rand((M, N), device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch_output = torch.mm(a.permute((1, 0)), b)
    triton_output = torch.empty_like(
        torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    print(a.stride(1), a.stride(0))
    triton_tn_kernel[grid](a, b, triton_output, M, N, K, a.stride(1), a.stride(0),
                           b.stride(0), b.stride(1), triton_output.stride(0), triton_output.stride(1), allow_tf32=False,
                           BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
    assert (torch.allclose(torch_output, triton_output, rtol=1e-3) is True)
