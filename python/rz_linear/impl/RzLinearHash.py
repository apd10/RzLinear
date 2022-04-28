def rz_linear_hash(v1: int, v0: int,
                   R7: int, R6: int, R5: int, R4: int,
                   R3: int, R2: int, R1: int, R0: int,
                   MOD: int) -> int:
    return (((v1 * R3 + v0 * R2 + R1) % R0) * R0 + ((v1 * R7 + v0 * R5 + R4) % R0)) % MOD
