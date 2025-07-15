# src/bch/bchcoder.py

import numpy as np
import logging
from typing import List

from sympy import Poly
from sympy.polys.domains import GF
from sympy.abc import x, alpha

from mathutils import order, power_dict, flatten_frac

log = logging.getLogger(__name__)

class BchCoder:
    """
    BCH encoder/decoder for binary (n, k, d) codes over GF(2).
    Corrects up to t = ⌊(d−1)/2⌋ errors via Berlekamp–Massey + Chien search.
    """

    def __init__(
        self,
        n: int,
        b: int,
        d: int,
        irr_poly: Poly,
        g_poly: Poly,
        q: int = 2
    ):
        # parameter checks
        if not all(isinstance(v, int) and v > 0 for v in (n, b, d)):
            raise ValueError("n, b, and d must be positive integers")
        if b + d - 1 > n:
            raise ValueError("Designed distance out of range: b + d - 1 must ≤ n")

        self.n, self.b, self.d, self.q = n, b, d, q
        self.r_poly = irr_poly.set_domain(GF(q))
        self.g_poly = g_poly.set_domain(GF(q))

        # derived parameters
        self.m = order(q, n)
        self.k = n - self.g_poly.degree()
        self.t = (d - 1) // 2

        log.info(f"BchCoder(n={n},k={self.k},d={d},t={self.t},m={self.m})")

    def encode(self, msg_poly: Poly) -> List[int]:
        msg = msg_poly.set_domain(GF(self.q))
        # shift message by n-k positions
        shifted = (msg * Poly(x ** (self.n - self.k), x, domain=GF(self.q))).set_domain(GF(self.q))
        remainder = shifted.rem(self.g_poly)
        code_poly = (shifted - remainder).set_domain(GF(self.q))
        coeffs = [int(c) for c in code_poly.all_coeffs()]
        # pad to length n
        if len(coeffs) < self.n:
            coeffs = [0]*(self.n - len(coeffs)) + coeffs
        return coeffs

    def decode(self, recv_poly: Poly) -> List[int]:
        recv = recv_poly.set_domain(GF(self.q))
        expr = recv.as_expr()

        # 1) syndromes S1..S2t
        S = []
        for i in range(1, 2*self.t + 1):
            val = expr.subs(x, alpha**i)
            S.append(Poly(val, alpha, domain=GF(self.q)).rem(self.r_poly))

        # 2) no errors?
        if all(si.is_zero for si in S):
            return self._extract_message(recv)

        # 3) BM → locator‐polynomial coefficients L[0..v]
        L = self._berlekamp_massey(S)

        # 4) Build locator as expr and Chien‐search
        Lambda_expr = sum(L[i].as_expr() * x**i for i in range(len(L)))
        errs = []
        for j in range(self.n):
            val = Lambda_expr.subs(x, alpha**(self.n - j))
            test = Poly(val, alpha, domain=GF(self.q))\
                    .rem(self.r_poly)\
                    .set_domain(GF(self.q))
            if test.is_zero:
                errs.append(j)

        # 5) flip bits in recv
        bits = [int(c) for c in recv.all_coeffs()]
        bits = [0]*(self.n-len(bits)) + bits
        for j in errs:
            idx = (self.n - 1) - j
            bits[idx] ^= 1

        # 6) extract data
        return bits[:self.k]


    def _berlekamp_massey(self, S: List[Poly]) -> List[Poly]:
        C = [Poly(1, alpha, domain=GF(self.q))]  # connection polynomial
        B = [Poly(1, alpha, domain=GF(self.q))]
        L = 0     # current degree
        m = 1     # step
        b = Poly(1, alpha, domain=GF(self.q))

        for n_i in range(len(S)):
            # discrepancy d = S[n] + sum_{i=1..L} C[i]*S[n-i]
            d = S[n_i]
            for i in range(1, L+1):
                d = (d + C[i] * S[n_i - i]).set_domain(GF(self.q)).rem(self.r_poly)
            if d.is_zero:
                m += 1
            else:
                T = C.copy()
                inv_b = b.invert(self.r_poly)
                f = (d * inv_b).set_domain(GF(self.q)).rem(self.r_poly)

                # C = C - f * x^m * B
                needed = m + len(B)
                if len(C) < needed:
                    C += [Poly(0, alpha, domain=GF(self.q))]*(needed - len(C))
                for j in range(len(B)):
                    C[j+m] = (C[j+m] + B[j]*f).set_domain(GF(self.q)).rem(self.r_poly)

                if 2*L <= n_i:
                    B = T
                    b = d
                    L = n_i + 1 - L
                    m = 1
                else:
                    m += 1

        # trim trailing zeros
        while len(C) > 1 and C[-1].is_zero:
            C.pop()
        return C

    def _extract_message(self, code: Poly) -> List[int]:
        coeffs = [int(c) for c in code.all_coeffs()]
        if len(coeffs) < self.n:
            coeffs = [0]*(self.n - len(coeffs)) + coeffs
        # return last k bits (highest-order)
        return coeffs[:self.k]

