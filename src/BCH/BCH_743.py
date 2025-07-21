import numpy as np
import numba
from numba import cuda, uint8
# Code parameters
n = 7        # code length
k = 4        # message length
t = 1        # corrects up to 1 error
deg = n - k  # degree of generator = 3

# Generator polynomial g(x) = x^3 + x + 1 → coefficients [1,1,0,1] for x^0…x^3
g = np.array([1, 1, 0, 1], dtype=np.uint8)

@numba.njit
def encode_stream(msg: np.ndarray) -> np.ndarray:
    """
    msg: 1D array of 0/1, length multiple of k=4
    returns: 1D array of 0/1, length = (len(msg)//4)*7
    """
    blocks = msg.size // k
    cw = np.empty(blocks * n, dtype=np.uint8)
    for i in range(blocks):
        # slice off the next 4‐bit block
        block = msg[i*k : i*k + k]
        # call your jitted encode_nb
        codeword = encode_nb(block)
        # write it out
        cw[i*n : i*n + n] = codeword
    return cw

@numba.njit
def decode_stream(cw: np.ndarray) -> np.ndarray:
    """
    cw: 1D array of 0/1, length multiple of n=7
    returns: 1D array of 0/1, length = (len(cw)//7)*4
    """
    blocks = cw.size // n
    msg = np.empty(blocks * k, dtype=np.uint8)
    for i in range(blocks):
        # slice off the next 7‐bit codeword
        block = cw[i*n : i*n + n]
        # call your jitted decode_nb
        decoded = decode_nb(block)
        # write the recovered 4-bits
        msg[i*k : i*k + k] = decoded
    return msg

@numba.njit
def _syndrome(cw: np.ndarray) -> bool:
    """
    Return True iff cw (length 7) is a valid codeword,
    i.e. cw(x) % g(x) == 0 under GF(2) polynomial division.
    """
    # Work on a copy so we don’t destroy cw
    T = cw.copy()
    # Long division: for i = 6 down to 3
    for i in range(n-1, deg-1, -1):
        if T[i] == 1:
            # subtract g(x) * x^(i-deg)
            for j in range(deg+1):       # j=0..3
                T[i-deg + j] ^= g[j]
    # If remainder T[0..2] are all zero, it divides cleanly
    for j in range(deg):
        if T[j] != 0:
            return False
    return True

@numba.njit
def encode_nb(msg: np.ndarray) -> np.ndarray:
    """
    Systematic encode for BCH(7,4,3):
    msg: uint8[4] with bits m[0]..m[3]
    returns: uint8[7] codeword [r0,r1,r2,m0,m1,m2,m3]
    """
    # 1) Build shifted message with 3 zero parity positions
    B = np.zeros(n, dtype=np.uint8)
    for i in range(k):
        B[i + deg] = msg[i]

    # 2) Compute remainder of B(x) divided by g(x)
    T = B.copy()
    for i in range(n-1, deg-1, -1):
        if T[i] == 1:
            for j in range(deg+1):
                T[i-deg + j] ^= g[j]

    # 3) Assemble systematic codeword: [r0,r1,r2,m0..m3]
    cw = np.empty(n, dtype=np.uint8)
    for j in range(deg):
        cw[j] = T[j]
    for i in range(k):
        cw[i + deg] = msg[i]
    return cw

@numba.njit
def decode_nb(cw: np.ndarray) -> np.ndarray:
    """
    Brute-force single-error decode for BCH(7,4,3):
    cw: uint8[7] received codeword
    returns: uint8[4] corrected message bits
    """
    # 0) Quick check: if already valid, just slice off the message
    if _syndrome(cw):
        msg = np.empty(k, dtype=np.uint8)
        for i in range(k):
            msg[i] = cw[i + deg]
        return msg

    # 1) Try flipping each single bit
    for e in range(n):
        cw2 = cw.copy()
        cw2[e] ^= 1
        if _syndrome(cw2):
            msg = np.empty(k, dtype=np.uint8)
            for i in range(k):
                msg[i] = cw2[i + deg]
            return msg

    # 2) If we get here, too many errors – just return what we can
    msg = np.empty(k, dtype=np.uint8)
    for i in range(k):
        msg[i] = cw[i + deg]
    return msg

# --------------------
# Quick test
# --------------------
if __name__ == "__main__":
    msg = np.random.randint(0,2, size=100_000, dtype=np.uint8)
    # no padding needed since 100000 % 4 == 0

    # encode entire stream
    cw = encode_stream(msg)

    # inject errors anywhere in the coded stream
    cw_noisy = cw.copy()
    err_pos   = np.random.randint(0, cw.size)
    cw_noisy[err_pos] ^= 1

    # decode entire stream block-by-block
    decoded = decode_stream(cw_noisy)

    assert np.array_equal(decoded, msg), "Decoding failed!"
    print("✔ Full-stream encode/decode OK")