import torch
import os

import triton
import triton.language as tl

"""

here is the implementation of your triton kernel
note that the kernel name should be 'kernel' to maintain consistency of test.py the main.c and test.py, or you can motify them

"""

@triton.jit
def kernel(
    A, B,
    M, N,
    param_hor, param_ver,
    #半径大小
    R,
    #分块信息
    BLOCK_SIZE_M: tl.constexpr, #==16
    BLOCK_SIZE_N: tl.constexpr #==16
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M-2*16, BLOCK_SIZE_M) 
    num_pid_n = tl.cdiv(N-2*16, BLOCK_SIZE_N)
    pid_m = (pid // num_pid_n) + 1
    pid_n = (pid % num_pid_n) + 1

    offset_16 = tl.arange(0, BLOCK_SIZE_M)

    para_h_ptrs = param_hor + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_h_data = tl.load(para_h_ptrs)

    para_v_ptrs = param_ver + ((offset_16[:,None])*BLOCK_SIZE_N + offset_16[None,:])
    para_v_data = tl.load(para_v_ptrs)

    a_brick_offset = (pid_m*(N//16) + pid_n)*16*16

    a_ptrs = A + a_brick_offset + ((offset_16[:,None])*16 + offset_16[None,:])
    a_data = tl.load(a_ptrs)

    r1 = tl.dot(a_data, para_h_data)
    r2 = tl.dot(para_v_data, a_data)
    accumulator = r1+r2
    halo_result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    halo_result = accumulator + halo_result
    b_ptrs = B + a_brick_offset + ((offset_16[:,None])*16 + offset_16[None,:])
    tl.store(b_ptrs, halo_result)