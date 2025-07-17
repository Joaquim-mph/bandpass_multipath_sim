import numpy as np
from numba import cuda, uint8
import math

n   = 15
k   = 7
deg = n - k

# Generator polynomial coefficients for g(x) = x^8 + x^7 + x^6 + x^4 + x^2 + 1
g_host = np.array([1,0,1,0,1,0,1,1,1], dtype=np.uint8)

# Precompute all 1- and 2-bit error syndromes
from itertools import combinations

def bch_syndrome_table():
    syndromes = {}
    def div(poly):
        rem = poly[:]
        for i in range(n-1, deg-1, -1):
            if rem[i]:
                for j in range(deg+1):
                    rem[i-deg+j] ^= g_host[j]
        return tuple(rem[:deg])

    for i in range(n):
        e = [0]*n
        e[i] = 1
        s = div(e)
        syndromes.setdefault(s, []).append([i])

    for i, j in combinations(range(n), 2):
        e = [0]*n
        e[i] = 1
        e[j] = 1
        s = div(e)
        syndromes.setdefault(s, []).append([i, j])

    return syndromes

syndrome_map = bch_syndrome_table()
syndrome_keys = list(syndrome_map.keys())

@cuda.jit(device=True)
def encode_block_fast(msg_blk, out_cw, g):
    B = cuda.local.array(n, uint8)
    for i in range(n):
        B[i] = 0
    for i in range(k):
        B[i + deg] = msg_blk[i]

    T = cuda.local.array(n, uint8)
    for i in range(n):
        T[i] = B[i]
    for i in range(n-1, deg-1, -1):
        if T[i]:
            for j in range(deg+1):
                T[i-deg + j] ^= g[j]

    for j in range(deg):
        out_cw[j] = T[j]
    for i in range(k):
        out_cw[i + deg] = msg_blk[i]

@cuda.jit(device=True)
def compute_syndrome(cw_blk, g, syn):
    for i in range(n):
        syn[i] = cw_blk[i]
    for i in range(n-1, deg-1, -1):
        if syn[i]:
            for j in range(deg+1):
                syn[i-deg + j] ^= g[j]

@cuda.jit(device=True)
def decode_block_fast(cw_blk, out_msg, g, syn_table_keys, syn_table_vals, syn_table_counts, syn_table_len):
    syn = cuda.local.array(n, uint8)
    compute_syndrome(cw_blk, g, syn)

    temp_blk = cuda.local.array(n, uint8)
    matched = False
    for i in range(syn_table_len):
        match = True
        for j in range(deg):
            if syn[j] != syn_table_keys[i, j]:
                match = False
                break
        if match:
            count = syn_table_counts[i]
            for m in range(count):
                for t in range(n):
                    temp_blk[t] = cw_blk[t]  # copy original
                for j in range(2):
                    err_pos = syn_table_vals[i, m, j]
                    if err_pos != 255:
                        temp_blk[err_pos] ^= 1
                # check if syndrome is now zero
                temp_syn = cuda.local.array(n, uint8)
                compute_syndrome(temp_blk, g, temp_syn)
                zero = True
                for j in range(deg):
                    if temp_syn[j]:
                        zero = False
                        break
                if zero:
                    for j in range(n):
                        cw_blk[j] = temp_blk[j]
                    matched = True
                    break
            break

    for i in range(k):
        out_msg[i] = cw_blk[i + deg]

@cuda.jit
def encode_stream_gpu(msg, cw, g):
    i = cuda.grid(1)
    blocks = msg.size // k
    if i < blocks:
        mblk = cuda.local.array(k, uint8)
        for j in range(k):
            mblk[j] = msg[i*k + j]
        out_cw = cuda.local.array(n, uint8)
        encode_block_fast(mblk, out_cw, g)
        base = i * n
        for j in range(n):
            cw[base + j] = out_cw[j]

@cuda.jit
def decode_stream_gpu(cw, msg, g, syn_table_keys, syn_table_vals, syn_table_counts, syn_table_len):
    i = cuda.grid(1)
    blocks = cw.size // n
    if i < blocks:
        cblk = cuda.local.array(n, uint8)
        for j in range(n):
            cblk[j] = cw[i*n + j]
        out_msg = cuda.local.array(k, uint8)
        decode_block_fast(cblk, out_msg, g, syn_table_keys, syn_table_vals, syn_table_counts, syn_table_len)
        base = i * k
        for j in range(k):
            msg[base + j] = out_msg[j]


if __name__ == "__main__":
    total_bits = 100_000
    padded_bits = ((total_bits + k - 1) // k) * k
    pad_len = padded_bits - total_bits

    msg_host = np.random.randint(0,2,size=total_bits).astype(np.uint8)
    if pad_len > 0:
        msg_host = np.concatenate([msg_host, np.zeros(pad_len, dtype=np.uint8)])

    blocks   = padded_bits // k

    msg_dev = cuda.to_device(msg_host)
    cw_dev  = cuda.device_array(blocks * n, dtype=np.uint8)
    g_dev   = cuda.to_device(g_host)

    threads = 256
    grid    = math.ceil(blocks / threads)
    encode_stream_gpu[grid, threads](msg_dev, cw_dev, g_dev)

    cw = cw_dev.copy_to_host()
    cw_clean = cw.copy()
    rng = np.random.default_rng()
    for b in range(blocks):
        pos1, pos2 = rng.choice(n, size=2, replace=False)
        cw[b*n + pos1] ^= 1
        cw[b*n + pos2] ^= 1
    cw_dev = cuda.to_device(cw)

    # Prepare syndrome table for GPU
    max_entries = len(syndrome_keys)
    syn_keys = np.zeros((max_entries, deg), dtype=np.uint8)
    syn_vals = np.full((max_entries, 4, 2), 255, dtype=np.uint8)  # Up to 4 patterns
    syn_counts = np.zeros(max_entries, dtype=np.uint8)

    for i, key in enumerate(syndrome_keys):
        syn_keys[i] = key
        for j, pattern in enumerate(syndrome_map[key][:4]):
            syn_vals[i, j, :len(pattern)] = pattern
        syn_counts[i] = len(syndrome_map[key][:4])

    syn_keys_dev   = cuda.to_device(syn_keys)
    syn_vals_dev   = cuda.to_device(syn_vals)
    syn_counts_dev = cuda.to_device(syn_counts)

    msg_out_dev = cuda.device_array(blocks * k, dtype=np.uint8)
    decode_stream_gpu[grid, threads](cw_dev, msg_out_dev, g_dev, syn_keys_dev, syn_vals_dev, syn_counts_dev, len(syndrome_keys))

    decoded = msg_out_dev.copy_to_host()
    decoded = decoded[:total_bits]

    # Compare block-by-block
    failures = 0
    failed_blocks = []
    for b in range(blocks):
        m_ref = msg_host[b*k:b*k+k]
        m_out = decoded[b*k:b*k+k]
        if not np.array_equal(m_ref, m_out):
            failures += 1
            if len(failed_blocks) < 10:
                orig_cw = cw_clean[b*n:b*n+n].tolist()
                corr_cw = cw[b*n:b*n+n].tolist()
                failed_blocks.append((b, m_ref.tolist(), m_out.tolist(), orig_cw, corr_cw))

    print(f"✔ Total blocks: {blocks}, Correctly decoded: {blocks - failures}, Failures: {failures}")
    for b, ref, out, clean, err in failed_blocks:
        print(f"Block {b} FAILED:\n  Expected: {ref}\n  Decoded : {out}\n  Codeword: {clean}\n  WithErrs: {err}")

    assert failures == 0, "❌ Some blocks failed to decode correctly."
