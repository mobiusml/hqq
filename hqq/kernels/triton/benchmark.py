import torch
from torch import uint8, float16, Tensor
import numpy as np

try:
	from hqq.kernels.triton.dequant import *
except:
	pass

compute_dtype = torch.float16
device        = 'cuda'

def torch_dequantize_4bit_u8_cat(W_q: Tensor, zeros:Tensor, scales:Tensor, dtype=float16) -> Tensor:  # uint8/2 > uint8
    W_r = torch.cat([(W_q & 0b11110000) >> 4, W_q & 0b00001111], axis=0) 
    W_r = (W_r - zeros)*scales
    return W_r

def torch_dequantize_4bit_u8(W_q: Tensor, zeros:Tensor, scales:Tensor, dtype=float16) -> Tensor:  # uint8/2 > uint8
    _step = W_q.shape[0]
    W_r   = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

    W_r[:_step] = (W_q & 0xF0) >> 4
    W_r[_step:] = (W_q & 0x0F)
    W_r         = (W_r - zeros)*scales

    return W_r

def torch_dequantize_4bit_u8_interleaved(W_q: Tensor, zeros:Tensor, scales:Tensor, dtype=float16) -> Tensor:# uint8/2 > uint8
    _step = W_q.shape[0]
    W_r   = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

    W_r[::2, ...]  = (W_q & 0xF0) >> 4
    W_r[1::2, ...] = (W_q & 0x0F)
    W_r            = (W_r - zeros)*scales
    return W_r

###############################################################################################################
import time
import numpy as np 
def time_it(fct, runs=2000):
    t = []
    for _ in range(runs):
        t1 = time.time()
        _  = fct()
        torch.cuda.synchronize()
        t2 = time.time()
        t.append(t2-t1)

    return np.mean(t[-int(runs/2):]) #with warm-up


def eval(fct_torch, fct_compile, fct_triton, input_shape, meta_shape, step, compute_dtype=torch.float16, device='cuda', tol=1e-3, precision=6):

	W_q    = torch.randint(0, 2**8, (int(input_shape[0]/step), input_shape[1]), device=device).to(torch.uint8).contiguous()
	if(meta_shape is not None):
		
		zeros  = torch.randn(meta_shape, device=device, dtype=compute_dtype).contiguous()
		scales = torch.randn(meta_shape, device=device, dtype=compute_dtype).contiguous()

		# zeros  = torch.rand(meta_shape, device=device, dtype=compute_dtype).contiguous()*(8/step)
		# scales = torch.rand(meta_shape, device=device, dtype=compute_dtype).contiguous()/100.


		inputs = {'W_q':W_q, 'zeros':zeros, 'scales':scales}
	else:
		inputs = {'W_q':W_q}

	time_torch = time_it(lambda: fct_torch(**inputs))
	y_torch    = fct_torch(**inputs)

	try:
		time_compile = time_it(lambda: fct_compile(**inputs))
	except:
		time_compile = 1

	time_triton = time_it(lambda: fct_triton(**inputs))
	
	y_torch  = fct_torch(**inputs)
	y_triton = fct_triton(**inputs)
	assert torch.abs(y_triton - y_torch).max() <= tol

	out = {'time_torch': time_torch, 'time_compile':time_compile, 'time_triton':time_triton, 
			'speedup_vs_torch':time_torch/time_triton, 'speedup_vs_compile':time_compile/time_triton}

	return {k: np.round(out[k], precision) for k in out}

##################################################################################################################
fct_torch   = torch_dequantize_4bit_u8_cat
fct_compile = torch.compile(torch_dequantize_4bit_u8_cat)
fct_triton  = dequantize_4bit_u8
step        = 2

# fct_torch   = torch_dequantize_4bit_u8
# fct_compile = torch.compile(torch_dequantize_4bit_u8)
# fct_triton  = dequantize_4bit_u8
# step        = 2


# fct_torch   = torch_dequantize_4bit_u8_interleaved
# fct_compile = torch.compile(torch_dequantize_4bit_u8_interleaved)
# fct_triton  = dequantize_4bit_u8_interleaved
# step        = 2


_SHAPES      = [[2048, 2048], [4096, 4096], [4096, 11008], [11008, 4096], [4096*2, 4096*2], [4096*4, 4096*4]] 
_GROUP_SIZES = [-1, 128]
_AXIS        = [1]


record = {}
for shape in _SHAPES: 
	for group_size in _GROUP_SIZES:
		for axis in _AXIS:
			if(axis==0):
				if(group_size==-1):
					input_shape = [shape[0], shape[1]] 
				else:
					input_shape = [group_size , int(np.prod(shape)/group_size)] 

				meta_shape = [1, input_shape[1 - axis]]

			else:
				if(group_size==-1):
					input_shape = [shape[0], shape[1]] 
				else:
					input_shape = [int(np.prod(shape)/group_size), group_size] 

				meta_shape = [input_shape[1 - axis], 1]


			key = 'W_r.shape_' + str(shape) + '_meta.shape_' + str(meta_shape)
			if(key not in record):
				record[key] = eval(fct_torch, fct_compile, fct_triton, input_shape, meta_shape=meta_shape, step=step, compute_dtype=torch.float16, device='cuda')
				print(key, record[key])


##############################################################################################################################################
# #4090 - cat (axis=1, group_sizes: [-1, 128])
# W_r.shape_[2048, 2048]_meta.shape_[2048, 1] {'time_torch': 7.5e-05, 'time_compile': 0.000165, 'time_triton': 0.000128, 'speedup_vs_torch': 0.588466, 'speedup_vs_compile': 1.286206}
# W_r.shape_[2048, 2048]_meta.shape_[32768, 1] {'time_torch': 7.5e-05, 'time_compile': 0.000165, 'time_triton': 0.000124, 'speedup_vs_torch': 0.600181, 'speedup_vs_compile': 1.329857}
# W_r.shape_[4096, 4096]_meta.shape_[4096, 1] {'time_torch': 0.000158, 'time_compile': 0.00018, 'time_triton': 0.000125, 'speedup_vs_torch': 1.262339, 'speedup_vs_compile': 1.438768}
# W_r.shape_[4096, 4096]_meta.shape_[131072, 1] {'time_torch': 0.000149, 'time_compile': 0.000173, 'time_triton': 0.000141, 'speedup_vs_torch': 1.055983, 'speedup_vs_compile': 1.233233}
# W_r.shape_[4096, 11008]_meta.shape_[4096, 1] {'time_torch': 0.000566, 'time_compile': 0.000407, 'time_triton': 0.000244, 'speedup_vs_torch': 2.31357, 'speedup_vs_compile': 1.665378}
# W_r.shape_[4096, 11008]_meta.shape_[352256, 1] {'time_torch': 0.000567, 'time_compile': 0.000398, 'time_triton': 0.000253, 'speedup_vs_torch': 2.239489, 'speedup_vs_compile': 1.570403}
# W_r.shape_[11008, 4096]_meta.shape_[11008, 1] {'time_torch': 0.000575, 'time_compile': 0.000413, 'time_triton': 0.000239, 'speedup_vs_torch': 2.40242, 'speedup_vs_compile': 1.727646}
# W_r.shape_[11008, 4096]_meta.shape_[352256, 1] {'time_torch': 0.000567, 'time_compile': 0.000399, 'time_triton': 0.000253, 'speedup_vs_torch': 2.244949, 'speedup_vs_compile': 1.580323}
# W_r.shape_[8192, 8192]_meta.shape_[8192, 1] {'time_torch': 0.000944, 'time_compile': 0.000639, 'time_triton': 0.000296, 'speedup_vs_torch': 3.184381, 'speedup_vs_compile': 2.155654}
# W_r.shape_[8192, 8192]_meta.shape_[524288, 1] {'time_torch': 0.000945, 'time_compile': 0.000621, 'time_triton': 0.000323, 'speedup_vs_torch': 2.926559, 'speedup_vs_compile': 1.923009}
# W_r.shape_[16384, 16384]_meta.shape_[16384, 1] {'time_torch': 0.003903, 'time_compile': 0.002255, 'time_triton': 0.000877, 'speedup_vs_torch': 4.450631, 'speedup_vs_compile': 2.571125}
# W_r.shape_[16384, 16384]_meta.shape_[2097152, 1] {'time_torch': 0.00394, 'time_compile': 0.002251, 'time_triton': 0.000938, 'speedup_vs_torch': 4.198953, 'speedup_vs_compile': 2.399101}


#4090 - interleaved (axis=1, group_sizes: [-1, 128])
# W_r.shape_[2048, 2048]_meta.shape_[2048, 1] {'time_torch': 9.6e-05, 'time_compile': 9.3e-05, 'time_triton': 0.000121, 'speedup_vs_torch': 0.795039, 'speedup_vs_compile': 0.768918}
# W_r.shape_[2048, 2048]_meta.shape_[32768, 1] {'time_torch': 9.6e-05, 'time_compile': 0.000104, 'time_triton': 0.000137, 'speedup_vs_torch': 0.701944, 'speedup_vs_compile': 0.75407}
# W_r.shape_[4096, 4096]_meta.shape_[4096, 1] {'time_torch': 0.000166, 'time_compile': 0.000107, 'time_triton': 0.000123, 'speedup_vs_torch': 1.350956, 'speedup_vs_compile': 0.872094}
# W_r.shape_[4096, 4096]_meta.shape_[131072, 1] {'time_torch': 0.000156, 'time_compile': 0.000108, 'time_triton': 0.000139, 'speedup_vs_torch': 1.121344, 'speedup_vs_compile': 0.773584}
# W_r.shape_[4096, 11008]_meta.shape_[4096, 1] {'time_torch': 0.000635, 'time_compile': 0.000186, 'time_triton': 0.000243, 'speedup_vs_torch': 2.610359, 'speedup_vs_compile': 0.764772}
# W_r.shape_[4096, 11008]_meta.shape_[352256, 1] {'time_torch': 0.000634, 'time_compile': 0.000189, 'time_triton': 0.000265, 'speedup_vs_torch': 2.393628, 'speedup_vs_compile': 0.712524}
# W_r.shape_[11008, 4096]_meta.shape_[11008, 1] {'time_torch': 0.00064, 'time_compile': 0.000196, 'time_triton': 0.000238, 'speedup_vs_torch': 2.6908, 'speedup_vs_compile': 0.822527}
# W_r.shape_[11008, 4096]_meta.shape_[352256, 1] {'time_torch': 0.000635, 'time_compile': 0.000192, 'time_triton': 0.000253, 'speedup_vs_torch': 2.511099, 'speedup_vs_compile': 0.759116}
# W_r.shape_[8192, 8192]_meta.shape_[8192, 1] {'time_torch': 0.00099, 'time_compile': 0.000318, 'time_triton': 0.000298, 'speedup_vs_torch': 3.325448, 'speedup_vs_compile': 1.069538}
# W_r.shape_[8192, 8192]_meta.shape_[524288, 1] {'time_torch': 0.000987, 'time_compile': 0.000307, 'time_triton': 0.000319, 'speedup_vs_torch': 3.092883, 'speedup_vs_compile': 0.96158}
# W_r.shape_[16384, 16384]_meta.shape_[16384, 1] {'time_torch': 0.004152, 'time_compile': 0.000982, 'time_triton': 0.000875, 'speedup_vs_torch': 4.744525, 'speedup_vs_compile': 1.121539}
# W_r.shape_[16384, 16384]_meta.shape_[2097152, 1] {'time_torch': 0.004162, 'time_compile': 0.000993, 'time_triton': 0.000935, 'speedup_vs_torch': 4.452569, 'speedup_vs_compile': 1.062519}


#4090 - interleaved (axis=1, group_sizes: [-1, 128])
# W_r.shape_[2048, 2048]_meta.shape_[2048, 1] {'time_torch': 0.000106, 'time_compile': 0.000105, 'time_triton': 0.000121, 'speedup_vs_torch': 0.875686, 'speedup_vs_compile': 0.869686}
# W_r.shape_[2048, 2048]_meta.shape_[32768, 1] {'time_torch': 0.000102, 'time_compile': 0.000106, 'time_triton': 0.000123, 'speedup_vs_torch': 0.829703, 'speedup_vs_compile': 0.860832}
# W_r.shape_[4096, 4096]_meta.shape_[4096, 1] {'time_torch': 0.000181, 'time_compile': 0.000115, 'time_triton': 0.000131, 'speedup_vs_torch': 1.382916, 'speedup_vs_compile': 0.874598}
# W_r.shape_[4096, 4096]_meta.shape_[131072, 1] {'time_torch': 0.000166, 'time_compile': 0.000113, 'time_triton': 0.000142, 'speedup_vs_torch': 1.169523, 'speedup_vs_compile': 0.793801}
# W_r.shape_[4096, 11008]_meta.shape_[4096, 1] {'time_torch': 0.000644, 'time_compile': 0.000225, 'time_triton': 0.00024, 'speedup_vs_torch': 2.682891, 'speedup_vs_compile': 0.937589}
# W_r.shape_[4096, 11008]_meta.shape_[352256, 1] {'time_torch': 0.000644, 'time_compile': 0.000224, 'time_triton': 0.000244, 'speedup_vs_torch': 2.64197, 'speedup_vs_compile': 0.919503}
# W_r.shape_[11008, 4096]_meta.shape_[11008, 1] {'time_torch': 0.000652, 'time_compile': 0.000229, 'time_triton': 0.000238, 'speedup_vs_torch': 2.734554, 'speedup_vs_compile': 0.958651}
# W_r.shape_[11008, 4096]_meta.shape_[352256, 1] {'time_torch': 0.000649, 'time_compile': 0.000223, 'time_triton': 0.000243, 'speedup_vs_torch': 2.66917, 'speedup_vs_compile': 0.915093}
# W_r.shape_[8192, 8192]_meta.shape_[8192, 1] {'time_torch': 0.000997, 'time_compile': 0.000302, 'time_triton': 0.000299, 'speedup_vs_torch': 3.338445, 'speedup_vs_compile': 1.010275}
# W_r.shape_[8192, 8192]_meta.shape_[524288, 1] {'time_torch': 0.000992, 'time_compile': 0.000295, 'time_triton': 0.000311, 'speedup_vs_torch': 3.19397, 'speedup_vs_compile': 0.94875}
# W_r.shape_[16384, 16384]_meta.shape_[16384, 1] {'time_torch': 0.004283, 'time_compile': 0.000896, 'time_triton': 0.000866, 'speedup_vs_torch': 4.944466, 'speedup_vs_compile': 1.034025}
# W_r.shape_[16384, 16384]_meta.shape_[2097152, 1] {'time_torch': 0.004283, 'time_compile': 0.000947, 'time_triton': 0.000894, 'speedup_vs_torch': 4.792538, 'speedup_vs_compile': 1.059928}



