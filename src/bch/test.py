# Test harness for BCH encoder/decoder
# This code assumes you have BchCodeGenerator and BchCoder implemented as above,
# and available in the module `bch.bchcoder`.

import random
import numpy as np
from sympy import Poly
from sympy.abc import x, alpha
from bchcoder import BchCoder
from bchcodegenerator import BchCodeGenerator


def test_bch(n, b, d, num_tests=100):
    """
    Test BCH code with parameters (n, k, d) by encoding random messages,
    introducing up to t errors, and verifying correct decoding.
    """
    # 1. Generate BCH code
    gen = BchCodeGenerator(n, b, d)
    irr_poly, g_poly = gen.gen()
    coder = BchCoder(n, b, d, irr_poly, g_poly)
    
    k = coder.k
    t = coder.t
    
    print(f"Testing BCH({n},{k},{d}), t={t}, on {num_tests} random messages.")
    
    for _ in range(num_tests):
        # random message of length k
        msg_bits = [random.randint(0,1) for _ in range(k)]
        msg_poly = Poly(sum(bit * x**i for i, bit in enumerate(reversed(msg_bits))), x)
        
        # encode
        codeword = coder.encode(msg_poly)
        
        # choose error weight <= t
        error_positions = random.sample(range(n), random.randint(0, t))
        received = codeword.copy()
        for pos in error_positions:
            received[pos] ^= 1
        
        # decode
        recv_poly = Poly(sum(bit * x**i for i, bit in enumerate(reversed(received))), x)
        decoded = coder.decode(recv_poly)
        
        assert decoded == msg_bits, (
            f"Decoded message {decoded} != original {msg_bits} "
            f"for errors at positions {error_positions}"
        )
    print("All tests passed for BCH({},{},{})".format(n, k, d))

if __name__ == "__main__":
    # Test small BCH codes
    test_bch(n=7, b=1, d=3, num_tests=100)    # (7,4) single-error
    test_bch(n=15, b=1, d=5, num_tests=100)   # (15,11) double-error
    test_bch(n=15, b=1, d=7, num_tests=100)   # (15,5) triple-error
