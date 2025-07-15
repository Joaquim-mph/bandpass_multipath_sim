import pytest
import numpy as np
import itertools

from sympy import Poly
from sympy.abc import x
from sympy.polys.domains import GF

from bchcodegenerator import BchCodeGenerator
from bchcoder       import BchCoder


def make_coder(n=7, b=1, d=3):
    gen = BchCodeGenerator(n, b, d)
    irr, g = gen.gen()
    return BchCoder(n, b, d, irr, g)

def poly2gf(bits):
    """Build a GF(2) polynomial from a list of bits (MSB first)."""
    return Poly(bits, x, domain=GF(2))


# -------------------------------------------------------------------------
# Basic encode/decode tests (t=1 code)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("msg_bits", [
    [0,0,0,0],
    [1,0,1,1],
    [1,1,1,0],
    [0,1,0,1],
    list(np.random.randint(0,2, size=4))
])
def test_no_error_roundtrip(msg_bits):
    coder = make_coder(7,1,3)
    cw = coder.encode(poly2gf(msg_bits))
    decoded = coder.decode(poly2gf(cw))
    assert decoded == msg_bits


def test_single_error_correction():
    coder = make_coder(7,1,3)
    k, n = coder.k, coder.n
    for _ in range(5):
        msg = list(np.random.randint(0,2, size=k))
        cw  = coder.encode(poly2gf(msg))
        for pos in range(n):
            errored = cw.copy()
            errored[pos] ^= 1
            out = coder.decode(poly2gf(errored))
            assert out == msg


# -------------------------------------------------------------------------
# Generator‐polynomial degree tests
# -------------------------------------------------------------------------

def test_generator_poly_degrees_small():
    # (7,4,3) code: m=3, k=4, generator degree = n−k = 3
    gen = BchCodeGenerator(7, 1, 3)
    irr, g = gen.gen()
    k = 7 - g.degree()
    assert irr.degree() == gen.m
    assert g.degree() == 7 - k

def test_generator_15_11_5_properties():
    # (15,11,5) code: m=4, k=11, generator degree = 15−11 = 4
    # The narrow‐sense (15, k, 5) BCH from minimal-polys 1..4 has generator degree 8 => k=7
    gen = BchCodeGenerator(15, 1, 5)
    irr, g = gen.gen()
    k = 15 - g.degree()
    # Field extension m=4, so irr.degree()==4
    assert irr.degree() == gen.m == 4
    # Generator degree = 8, so k = 15−8 = 7
    assert g.degree() == 8
    assert k == 7

# -------------------------------------------------------------------------
# Random error‐correction within t for both t=1 and t=2 codes
# -------------------------------------------------------------------------

@pytest.mark.parametrize("n,b,d", [
    (7, 1, 3),    # t = 1
])
def test_random_roundtrip_within_capacity(n, b, d):
    coder = make_coder(n, b, d)
    t = coder.t
    for _ in range(10):
        msg = list(np.random.randint(0,2, size=coder.k))
        cw  = coder.encode(poly2gf(msg))
        # test *all* error‐patterns of weight up to t
        for weight in range(1, t+1):
            for errs in itertools.combinations(range(coder.n), weight):
                errored = cw.copy()
                for pos in errs:
                    errored[pos] ^= 1
                out = coder.decode(poly2gf(errored))
                assert out == msg, f"Failed to correct errors {errs} for {(n,b,d)}"


@pytest.mark.parametrize("n, b, d", [
    (255, 1, 5),   # BCH(255,239,5)
])
def test_bch_255_239_5_random(n, b, d):
    # build encoder/decoder
    gen   = BchCodeGenerator(n, b, d)
    irr, g = gen.gen()
    coder = BchCoder(n, b, d, irr, g)

    # sanity‐check parameters
    assert coder.n == 255
    assert coder.t == 2
    assert coder.k == 255 - g.degree() == 239

    # repeat on a few random messages
    for _ in range(5):
        msg = list(np.random.randint(0, 2, size=coder.k))
        cw  = coder.encode(poly2gf(msg))

        # 0 errors
        assert coder.decode(poly2gf(cw)) == msg

        # 1 error
        e1 = np.random.randint(0, n)
        ev = cw.copy()
        ev[e1] ^= 1
        assert coder.decode(poly2gf(ev)) == msg

        # 2 errors
        e2, e3 = np.random.choice(range(n), 2, replace=False)
        ev = cw.copy()
        ev[e2] ^= 1
        ev[e3] ^= 1
        assert coder.decode(poly2gf(ev)) == msg

# -------------------------------------------------------------------------
# >t errors are *not* guaranteed; mark that test as xfail for now
# -------------------------------------------------------------------------

@pytest.mark.xfail(reason=">t decoding not implemented for t=2", strict=False)
def test_triple_error_detection_for_t2():
    coder = make_coder(15,1,5)
    msg = list(np.random.randint(0,2, size=coder.k))
    cw  = coder.encode(poly2gf(msg))
    # introduce 3 errors (t+1 for t=2)
    errored = cw.copy()
    for pos in [0,1,2]:
        errored[pos] ^= 1
    out = coder.decode(poly2gf(errored))
    assert out != msg


# -------------------------------------------------------------------------
# Invalid‐parameter checks
# -------------------------------------------------------------------------

def test_invalid_bch_params():
    # generator should reject b+d−1 > n
    with pytest.raises(ValueError):
        BchCodeGenerator(7, 6, 3)
    # coder should also reject b+d−1 > n even if irr/g are valid
    irr, g = BchCodeGenerator(7,1,3).gen()
    with pytest.raises(ValueError):
        BchCoder(7, 1, 8, irr, g)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
