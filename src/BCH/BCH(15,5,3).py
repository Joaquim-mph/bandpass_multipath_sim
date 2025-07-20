import numpy as np
import galois
from functools import reduce

# Parameters for BCH(15,5,3)
m = 4
GF_ext = galois.GF(2**m)      # extension field for decoding
GF2    = galois.GF(2)         # binary field for encoding
alpha  = GF_ext.primitive_element
n      = 2**m - 1             # 15
t      = 3                    # correct up to 3 errors
k      = 5                    # data bits

# (1) Build generator polynomial g(x) = lcm of minimal polynomials φ₁, φ₃, φ₅
P1 = galois.Poly([1,0,0,1,1], field=GF2)  # φ₁(x) = x^4 + x + 1
P3 = galois.Poly([1,1,0,0,1], field=GF2)  # φ₃(x) = x^4 + x^3 + 1
P5 = galois.Poly([1,1,1],     field=GF2)  # φ₅(x) = x^2 + x + 1
generator = reduce(lambda A, B: galois.lcm(A, B), [P1, P3, P5])

# (2) Systematic encoder over GF2
def encode(message_bits):
    assert message_bits.shape == (k,)
    m_poly = galois.Poly(message_bits, field=GF2)
    xnk = galois.Poly.Degrees([n-k], coeffs=[1], field=GF2)
    shifted = m_poly * xnk
    remainder = shifted % generator
    c_poly = shifted + remainder
    c = np.zeros(n, dtype=np.uint8)
    for deg, coeff in zip(c_poly.degrees, c_poly.coeffs):
        c[n - 1 - deg] = int(coeff)
    return c

# (3) Decode via Peterson-Gorenstein-Zierler algorithm
def decode(received):
    assert received.shape == (n,)
    # Form received polynomial in GF_ext: coefficients degree-descending
    r_ext = GF_ext(received[::-1])
    r_poly = galois.Poly(r_ext, field=GF_ext)

    # Compute 2t syndromes
    S = [r_poly(alpha**i) for i in range(1, 2*t + 1)]
    if all(int(si) == 0 for si in S):
        return received.copy(), received[-k:].copy()

    # Try error weight u = t, t-1, ..., 1
    for u in range(t, 0, -1):
        # Build syndrome matrix M (u x u)
        M = GF_ext.Zeros((u, u))
        for i in range(u):
            for j in range(u):
                M[i, j] = S[i + j]
        # RHS vector b = [S_{u+1}, ..., S_{2u}] (negation is identity in GF(2^m))
        b = GF_ext.Zeros(u)
        for i in range(u):
            b[i] = S[u + i]
        # Solve M * lam_rev = b
        try:
            lam_rev = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            continue
        # Cast to GF_ext
        lam_rev = GF_ext(lam_rev)
        # Build error-locator polynomial Λ(x) = 1 + λ1 x + ... + λu x^u
        # galois.Poly takes coeffs in degree-descending order: [λu, ..., λ1, 1]
        coeffs = list(lam_rev[::-1]) + [GF_ext(1)]
        C = galois.Poly(coeffs, field=GF_ext)
        # Chien search: find roots => error positions
        errors = []
        for i in range(n):
            if C(alpha**i) == 0:
                errors.append(n - 1 - i)
        if len(errors) == u:
            break

    # Correct errors
    corrected = received.copy()
    for pos in errors:
        corrected[pos] ^= 1
    # Extract message bits
    message = corrected[-k:].copy()
    return corrected, message
# (4) Self-test
if __name__ == "__main__":
    np.random.seed(0)
    msg = np.random.randint(0, 2, size=k, dtype=np.uint8)
    cw  = encode(msg)
    errs = np.random.choice(n, size=t, replace=False)
    rx = cw.copy(); rx[errs] ^= 1

    rx_corr, msg_dec = decode(rx)
    print(f"Original codeword:     {cw}")
    print(f"Injected error pos:    {errs}")
    print(f"Received:              {rx}")
    print(f"Corrected:             {rx_corr}")
    print(f"Decoded message:       {msg_dec}")
    print(f"Original message:      {msg}")

    if np.array_equal(msg_dec, msg):
        print("✔ BCH(15,5,3) encode/decode successful!")
    else:
        print("❌ BCH(15,5,3) decode FAILED")
