import numpy as np
from sympy import Poly
from sympy.polys.domains import GF
from sympy.abc import x, alpha
from sympy import Expr
from typing import Dict, Tuple


def order(q: int, p: int) -> int:
    """
    Multiplicative order of q modulo p: smallest k>0 with q^k ≡ 1 (mod p).
    Raises ValueError if no order found.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    tx = q % p
    for k in range(1, p + 1):
        if tx == 1:
            return k
        tx = (tx * q) % p
    raise ValueError(f"No multiplicative order for q={q} mod p={p}")


def minimal_poly(exp: int, n: int, q: int, irr_poly: Poly) -> Poly:
    """
    Compute minimal polynomial of alpha**exp over GF(q), given the
    irreducible field polynomial `irr_poly` of degree m with n = q^m - 1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    seen = set()
    e = exp % n
    seen.add(e)
    poly = Poly(x - alpha**e, x)
    while True:
        e = (e * q) % n
        if e in seen:
            # close the cycle: extract constant‐term coefficients
            coeffs = []
            for c in poly.all_coeffs():
                c_red = (Poly(c, alpha).set_domain(GF(q)) % irr_poly)
                if c_red.degree() > 0:
                    raise ValueError("Minimal polynomial factor has unexpected degree")
                coeffs.append(int(c_red.nth(0)))
            return Poly(coeffs, x)
        seen.add(e)
        poly *= Poly(x - alpha**e, x)


def power_dict(n: int, irr_poly: Poly, q: int) -> Dict[Tuple[int, ...], int]:
    """
    Map each field element (as tuple of coeffs) to its exponent in GF(q^m).
    Only maps nonzero powers 1..n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    mapping: Dict[Tuple[int, ...], int] = {}
    for i in range(1, n + 1):
        elt = (Poly(alpha**i, alpha).set_domain(GF(q)) % irr_poly)
        mapping[tuple(elt.all_coeffs())] = i
    return mapping


def flatten_frac(expr: Expr, mod_poly: Poly, q: int,
                 pow_map: Dict[Tuple[int, ...], int]) -> Poly:
    """
    Simplify a GF(q) rational expression `expr = num/den` into a field element.
    Returns a Poly in x representing alpha^k mod `mod_poly`.
    """
    # Decompose into numerator/denominator
    num, den = expr.as_numer_denom()
    num_p = (Poly(num, alpha).set_domain(GF(q)) % mod_poly)
    den_p = (Poly(den, alpha).set_domain(GF(q)) % mod_poly)

    if den_p.is_zero:
        raise ZeroDivisionError("Division by zero in GF field")

    key_num = tuple(num_p.all_coeffs())
    key_den = tuple(den_p.all_coeffs())

    if key_num not in pow_map or key_den not in pow_map:
        raise ValueError(f"Element not in power map: {expr}")

    exp = (pow_map[key_num] - pow_map[key_den]) % (len(pow_map))
    return (Poly(alpha**exp, alpha).set_domain(GF(q)) % mod_poly)


def padding_encode(input_arr: np.ndarray, block_size: int) -> np.ndarray:
    """
    Zero-pad `input_arr` so its length is a multiple of `block_size`,
    then append one marker block (length = block_size) whose last
    `n` entries are 1 to record how many zeros were added.

    Parameters
    ----------
    input_arr
        1D array of bits (or any ints).
    block_size
        Must be >= 1.

    Returns
    -------
    padded_arr
        1D array of length L + n + block_size, where
        - n = (−L) mod block_size
        - the final `block_size` entries are zeros followed by `n` ones.
    """
    L = len(input_arr)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    # pad length so that (L + n) % block_size == 0
    n = (-L) % block_size

    # 1) pad input with n zeros
    padded = np.pad(input_arr, (0, n), constant_values=0)

    # 2) build marker block: [0 ... 0, 1 ... 1] length=block_size
    marker = np.concatenate((
        np.zeros(block_size - n, dtype=input_arr.dtype),
        np.ones(n,              dtype=input_arr.dtype),
    ))

    return np.concatenate((padded, marker))


def padding_decode(padded_arr: np.ndarray, block_size: int) -> np.ndarray:
    """
    Invert `padding_encode`, recovering the original array.

    Parameters
    ----------
    padded_arr
        Output of `padding_encode`.
    block_size
        Same as the encoder’s.

    Returns
    -------
    original_arr
        The first L entries, before padding.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if padded_arr.ndim != 1 or len(padded_arr) < block_size:
        raise ValueError("Array too short to contain a marker block")

    # Extract the marker block
    marker = padded_arr[-block_size:]

    # Number of padding zeros was encoded as number of 1’s in marker
    n = int(np.count_nonzero(marker))

    # Compute original length
    L = len(padded_arr) - block_size - n
    if L < 0:
        raise ValueError("Invalid padding or block_size")

    return padded_arr[:L]