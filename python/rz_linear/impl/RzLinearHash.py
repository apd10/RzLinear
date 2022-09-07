import triton


@triton.jit
def rz_linear_hash_tl(v1, v0, R7, R6, R5, R4, R3, R2, R1, R0, MOD):
    return (((v1 * R3 + v0 * R2 + R1) % R0) * R0 + ((v1 * R7 + v0 * R5 + R4) % R0)) % MOD


def rz_linear_hash(v1, v0, R7, R6, R5, R4, R3, R2, R1, R0, MOD):
    return (((v1 * R3 + v0 * R2 + R1) % R0) * R0 + ((v1 * R7 + v0 * R5 + R4) % R0)) % MOD
