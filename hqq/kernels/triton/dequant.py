#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

import triton
import triton.language as tl
import torch
from torch import uint8, float16, Tensor

def default_triton_config(n_cols):
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    return {'BLOCK_SIZE':BLOCK_SIZE, 'num_warps':num_warps}

#####################################################################################################################################
#4-bit 
@triton.jit
def dequantize_4bit_u8_axis0_kernel(Wq_ptr, zeros_ptr, scales_ptr, Wr_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE:tl.constexpr):
    row_idx     = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    n           = n_rows*n_cols

    Wq_offsets  = row_idx * input_row_stride + col_offsets   
    Wr_chunk1   = row_idx * output_row_stride + col_offsets  #[:step]
    Wr_chunk2   = Wr_chunk1 + n #[step:]

    mask        = col_offsets < n_cols
    
    ##################################
    #axis == 0
    sz_offset = Wq_offsets % n_cols
    scales_0  = tl.load(scales_ptr + sz_offset, mask=mask) 
    scales_1  = scales_0

    zeros_0   = tl.load(zeros_ptr + sz_offset, mask=mask)
    zeros_1   = zeros_0
    ##################################

    Wq_0 = tl.load(Wq_ptr + Wq_offsets, mask=mask)

    Wr_0 = (((Wq_0 & 0xF0) >> 4) - zeros_0)*scales_0
    Wr_1 = ((Wq_0 & 0x0F)        - zeros_1)*scales_1

    ##################################

    tl.store(Wr_ptr + Wr_chunk1, Wr_0, mask=mask) #[:step]
    tl.store(Wr_ptr + Wr_chunk2, Wr_1, mask=mask) #[step:]

@triton.jit
def dequantize_4bit_u8_axis1_kernel(Wq_ptr, zeros_ptr, scales_ptr, Wr_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE:tl.constexpr):
    row_idx     = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    n           = n_rows*n_cols

    Wq_offsets  = row_idx * input_row_stride + col_offsets   
    Wr_chunk1   = row_idx * output_row_stride + col_offsets  #[:step]
    Wr_chunk2   = Wr_chunk1 + n #[step:]

    mask        = col_offsets < n_cols
    
    #################
    #axis == 1
    sz_offset = Wr_chunk1 // n_cols
    scales_0  = tl.load(scales_ptr + sz_offset, mask=mask) 
    scales_1  = tl.load(scales_ptr + sz_offset + n_rows, mask=mask) 

    zeros_0 = tl.load(zeros_ptr + sz_offset, mask=mask) 
    zeros_1 = tl.load(zeros_ptr + sz_offset + n_rows, mask=mask) 
    #################

    Wq_0 = tl.load(Wq_ptr + Wq_offsets, mask=mask)

    Wr_0 = (((Wq_0 & 0xF0) >> 4) - zeros_0)*scales_0
    Wr_1 = ((Wq_0 & 0x0F)        - zeros_1)*scales_1

    ##################################

    tl.store(Wr_ptr + Wr_chunk1, Wr_0, mask=mask) #[:step]
    tl.store(Wr_ptr + Wr_chunk2, Wr_1, mask=mask) #[step:]


def dequantize_4bit_u8(W_q, zeros, scales, dtype=torch.float16):
    n_rows, n_cols = W_q.shape

    step = 2
    W_r  = torch.empty([step*n_rows, n_cols], dtype=dtype, device=W_q.device)

    if(zeros.shape[0]==1 and scales.shape[0]==1):
        dequant_4bit_kernel = dequantize_4bit_u8_axis0_kernel
    else:
        dequant_4bit_kernel = dequantize_4bit_u8_axis1_kernel

    dequant_4bit_kernel[(n_rows, )](
        W_q,
        zeros, 
        scales,
        W_r,
        W_q.stride(0),
        W_r.stride(0),
        n_rows,
        n_cols,
        **default_triton_config(n_cols),
    )
    return W_r


#Interleaved
@triton.jit
def dequantize_4bit_u8_axis0_interleaved_kernel(Wq_ptr, zeros_ptr, scales_ptr, Wr_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE:tl.constexpr):
    row_idx     = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    Wq_offsets  = row_idx * input_row_stride + col_offsets   
    Wr_chunk1   = row_idx * output_row_stride*2 + col_offsets #[::2]
    Wr_chunk2   = Wr_chunk1  + n_cols    #[1::2]

    mask        = col_offsets < n_cols
    
    ##################################
    #axis == 0
    sz_offset = Wq_offsets % n_cols
    scales_0  = tl.load(scales_ptr + sz_offset, mask=mask)
    scales_1  = scales_0

    zeros_0   = tl.load(zeros_ptr + sz_offset, mask=mask) 
    zeros_1   = zeros_0
    ##################################

    Wq_0 = tl.load(Wq_ptr + Wq_offsets, mask=mask)

    Wr_0 = (((Wq_0 & 0xF0) >> 4) - zeros_0)*scales_0
    Wr_1 = ((Wq_0 & 0x0F)        - zeros_1)*scales_1

    ##################################

    tl.store(Wr_ptr + Wr_chunk1, Wr_0, mask=mask) #[::2]
    tl.store(Wr_ptr + Wr_chunk2, Wr_1, mask=mask) #[1::2]


@triton.jit
def dequantize_4bit_u8_axis1_interleaved_kernel(Wq_ptr, zeros_ptr, scales_ptr, Wr_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE:tl.constexpr):
    row_idx     = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    Wq_offsets  = row_idx * input_row_stride + col_offsets   
    Wr_chunk1   = row_idx * output_row_stride*2 + col_offsets #[::2]
    Wr_chunk2   = Wr_chunk1  + n_cols    #[1::2]

    mask        = col_offsets < n_cols
    
    ##################################
    #axis == 1
    sz_offset = Wr_chunk1 // n_cols
    scales_0  = tl.load(scales_ptr + sz_offset, mask=mask) 
    scales_1  = tl.load(scales_ptr + sz_offset + 1, mask=mask) 

    zeros_0 = tl.load(zeros_ptr + sz_offset, mask=mask) 
    zeros_1 = tl.load(zeros_ptr + sz_offset + 1, mask=mask) 
    ##################################

    Wq_0 = tl.load(Wq_ptr + Wq_offsets, mask=mask)

    Wr_0 = (((Wq_0 & 0xF0) >> 4) - zeros_0)*scales_0
    Wr_1 = ((Wq_0 & 0x0F)        - zeros_1)*scales_1

    ##################################

    tl.store(Wr_ptr + Wr_chunk1, Wr_0, mask=mask) #[::2]
    tl.store(Wr_ptr + Wr_chunk2, Wr_1, mask=mask) #[1::2]


def dequantize_4bit_u8_interleaved(W_q, zeros, scales, dtype=torch.float16):
    n_rows, n_cols = W_q.shape
    
    step = 2
    W_r  = torch.empty([step*n_rows, n_cols], dtype=dtype, device=W_q.device)

    if(zeros.shape[0]==1 and scales.shape[0]==1):
        dequant_4bit_kernel = dequantize_4bit_u8_axis0_interleaved_kernel
    else:
        dequant_4bit_kernel = dequantize_4bit_u8_axis1_interleaved_kernel

    dequant_4bit_kernel[(n_rows, )](
        W_q,
        zeros, 
        scales,
        W_r,
        W_q.stride(0),
        W_r.stride(0),
        n_rows,
        n_cols,
        **default_triton_config(n_cols),
    )
    return W_r


#####################################################################################################################################