import numpy as np
from numba import cuda, uint8
import math

# BCH(7,4,3) parameters
n = 7
k = 4
deg = n - k  # 3

@cuda.jit(device=True)
def encode_block_fast(msg_blk, out_cw):
    # unpack message bits
    m0 = msg_blk[0]; m1 = msg_blk[1]; m2 = msg_blk[2]; m3 = msg_blk[3]
    # remainder of x^3*m(x) mod g(x)=x^3+x+1
    p0 = m0 ^ m2 ^ m3
    p1 = m0 ^ m1 ^ m2
    p2 = m1 ^ m2 ^ m3
    out_cw[0] = p0; out_cw[1] = p1; out_cw[2] = p2
    out_cw[3] = m0; out_cw[4] = m1; out_cw[5] = m2; out_cw[6] = m3

@cuda.jit(device=True)
def decode_block_fast(cw_blk, out_msg):
    # cyclic syndrome s(x)=c(x) mod g(x), contributions for each bit
    s = uint8(0)
    if cw_blk[0]: s ^= uint8(1)
    if cw_blk[1]: s ^= uint8(2)
    if cw_blk[2]: s ^= uint8(4)
    if cw_blk[3]: s ^= uint8(3)
    if cw_blk[4]: s ^= uint8(6)
    if cw_blk[5]: s ^= uint8(7)
    if cw_blk[6]: s ^= uint8(5)
    # correct single-bit error
    if s != 0:
        if s == 1: idx = 0
        elif s == 2: idx = 1
        elif s == 4: idx = 2
        elif s == 3: idx = 3
        elif s == 6: idx = 4
        elif s == 7: idx = 5
        else:           idx = 6  # s==5
        cw_blk[idx] ^= uint8(1)
    # extract systematic message bits
    for i in range(k):
        out_msg[i] = cw_blk[i + deg]

@cuda.jit
def encode_stream_gpu(msg, cw):
    i = cuda.grid(1)
    blocks = msg.size // k
    if i < blocks:
        mblk = cuda.local.array(k, uint8)
        for j in range(k): mblk[j] = msg[i*k + j]
        out_cw = cuda.local.array(n, uint8)
        encode_block_fast(mblk, out_cw)
        base = i * n
        for j in range(n): cw[base + j] = out_cw[j]

@cuda.jit
def decode_stream_gpu(cw, msg):
    i = cuda.grid(1)
    blocks = cw.size // n
    if i < blocks:
        cblk = cuda.local.array(n, uint8)
        for j in range(n): cblk[j] = cw[i*n + j]
        out_msg = cuda.local.array(k, uint8)
        decode_block_fast(cblk, out_msg)
        base = i * k
        for j in range(k): msg[base + j] = out_msg[j]


if __name__ == "__main__":
    total_bits = 100_000
    assert total_bits % k == 0
    msg_host = np.random.randint(0, 2, size=total_bits).astype(np.uint8)
    blocks = total_bits // k

    # allocate GPU buffers
    msg_dev = cuda.to_device(msg_host)
    cw_dev = cuda.device_array(blocks * n, dtype=np.uint8)
    msg_out_dev = cuda.device_array(blocks * k, dtype=np.uint8)

    # launch config
    threads = 512
    grid = (blocks + threads - 1) // threads

    encode_stream_gpu[grid, threads](msg_dev, cw_dev)

    # inject one error every 7 bits
    cw = cw_dev.copy_to_host()
    for pos in range(0, cw.size, 7):
        cw[pos] ^= 1
    cw_dev = cuda.to_device(cw)

    decode_stream_gpu[grid, threads](cw_dev, msg_out_dev)

    decoded = msg_out_dev.copy_to_host()
    assert np.array_equal(decoded, msg_host)


    print("✔ Fast CUDA BCH encode/decode OK!")

