import logging
from typing import Tuple

import numpy as np
from sympy import Poly, ZZ, lcm
from sympy.polys.domains import GF
from sympy.polys.galoistools import gf_irreducible, gf_irreducible_p
from sympy.abc import alpha

from mathutils import order, minimal_poly, power_dict

log = logging.getLogger(__name__)

class BchCodeGenerator:
    """
    Generates the irreducible field polynomial and BCH generator polynomial
    for a binary BCH code of length n, starting root exponent b, designed
    distance d.
    """
    def __init__(self, n: int, b: int, d: int, q: int = 2):
        if not all(isinstance(v, int) and v > 0 for v in (n, b, d)):
            raise ValueError("n, b, and d must be positive integers")
        if b + d - 1 > n:
            raise ValueError("b + d - 1 must not exceed n")
        self.n, self.b, self.d, self.q = n, b, d, q
        self.m = order(self.q, self.n)
        log.info(f"Initialized BCH gen n={n}, b={b}, d={d}, q={q}, m={self.m}")

    def generate_irreducible_poly(self) -> Poly:
        """Find an irreducible degree-m polynomial over GF(q) with field-size ≥ n."""
        # Candidate 1: x^m + x + 1
        irr = Poly(alpha**self.m + alpha + 1, alpha).set_domain(GF(self.q))
        size = len(power_dict(self.n, irr, self.q)) if gf_irreducible_p(irr.all_coeffs(), self.q, ZZ) else 0
        log.debug(f"Test irr_poly size={size}: {irr}")
        # If too small or not irreducible, hunt for one using sympy's gf_irreducible
        while size < self.n:
            coeffs = [int(c) for c in gf_irreducible(self.m, self.q, ZZ)]
            irr = Poly(coeffs, alpha).set_domain(GF(self.q))
            size = len(power_dict(self.n, irr, self.q))
            log.debug(f"Retry irr_poly size={size}: {irr}")
        log.info(f"Selected irreducible poly: {irr}")
        return irr

    def compute_generator_poly(self, irr_poly: Poly) -> Poly:
        """Compute the BCH generator polynomial g(x) = lcm( minimal_poly(b..b+d-2) )."""
        g = None
        for exp in range(self.b, self.b + self.d - 1):
            mp = minimal_poly(exp, self.n, self.q, irr_poly)
            g = mp if g is None else lcm(g, mp)
        g = g.trunc(self.q)
        log.info(f"Computed generator poly g(x): {g}")
        return g

    def gen(self) -> Tuple[Poly, Poly]:
        """Return (irreducible_field_poly, generator_poly)."""
        irr = self.generate_irreducible_poly()
        g   = self.compute_generator_poly(irr)
        return irr, g
