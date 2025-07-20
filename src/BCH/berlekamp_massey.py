import numpy as np
import galois

def berlekamp_massey(syndromes, field_order=2, m=4):
    """
    Berlekamp-Massey algorithm to find the connection polynomial C(x)
    that generates the syndrome sequence over GF(field_order**m).
    Returns (C, L) where C is a galois.Poly and L is its degree.
    """
    # Finite field GF(q^m)
    GF = galois.GF(field_order**m)
    s = GF(syndromes)

    # Initialize polynomials
    C = galois.Poly([1], field=GF)  # connection polynomial
    B = galois.Poly([1], field=GF)  # backup polynomial
    L = 0                           # current LFSR length
    m_ = 1                          # steps since last update
    b = GF(1)                       # last nonzero discrepancy

    n = len(s)
    for N in range(n):
        # Build coefficient list in ascending order length L+1
        c_desc = C.coeffs             # degree-descending
        c_asc = c_desc[::-1]          # ascending
        if len(c_asc) < L+1:
            # Pad with zeros up to length L+1
            c_asc = np.concatenate([c_asc, GF.Zeros(L+1 - len(c_asc))])

        # Compute discrepancy d
        d = s[N]
        for i in range(1, L+1):
            d += c_asc[i] * s[N - i]

        if d == 0:
            m_ += 1
        else:
            # Copy current C to T
            T = galois.Poly(C.coeffs, field=GF)
            factor = d / b
            # C(x) = C(x) - (d/b) * x^m_ * B(x)
            shift_B = B * galois.Poly.Degrees([m_], coeffs=[1], field=GF)
            C = C - factor * shift_B

            if 2 * L <= N:
                # Update parameters when length increases
                L_new = N + 1 - L
                B = T
                b = d
                L = L_new
                m_ = 1
            else:
                m_ += 1

    return C, L

if __name__ == "__main__":
    # Example usage and test
    syndromes = [1, 0, 1, 0, 1]
    C, L = berlekamp_massey(syndromes, field_order=2, m=4)
    print(f"Connection polynomial: {C}")
    print(f"Degree (L): {L}")
