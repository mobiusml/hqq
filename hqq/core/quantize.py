#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################

import torch
import numpy as np 
from tqdm import tqdm

import gc
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

#Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal(tensor, scale, zero, min_max, axis=0, device='cuda', opt_params={'lp_norm':0.7, 'beta':1e1, 'kappa':1.01, 'iters':20}, verbose=False):
	lp_norm, beta, kappa, iters = opt_params['lp_norm'], opt_params['beta'], opt_params['kappa'], opt_params['iters']

	dtype  = torch.float16 if (device=='cuda') else torch.float32
	W_f    = tensor.to(dtype).to(device)
	scale  = scale.to(dtype).to(device)
	zero   = zero.to(dtype).to(device)

	if(lp_norm==1):
		shrink_op = lambda x, beta: torch.sign(x)*torch.nn.functional.relu(torch.abs(x) - 1./beta) 
	else:
		shrink_op = lambda x, beta,p=lp_norm: torch.sign(x)*torch.nn.functional.relu(torch.abs(x) - (1./beta)*torch.pow(torch.abs(x), p-1))

	best_error = 1e4
	for i in range(iters):
		W_q   = torch.round(W_f*scale + zero).clamp(min_max[0], min_max[1])
		W_r   = (W_q - zero)/scale
		W_e   = shrink_op(W_f - W_r, beta)
		zero  = torch.mean(W_q - (W_f - W_e)*scale, axis=axis, keepdim=True)
		beta *= kappa

		current_error = float(torch.abs(W_f - W_r).mean())
		if(verbose): 
			print(i, np.round(current_error, 6))
		if(current_error < best_error):
			best_error = current_error
		else:
			break

	scale = scale.to(tensor.device)
	zero  = zero.to(tensor.device)
	del W_f, W_q, W_r, W_e
	torch.cuda.empty_cache()

	return scale, zero 

#SGD solver  || W - dequantize(quantize(W))||_1 (p=1 only)
def optimize_weights_autograd(tensor, scale, zero, min_max, axis=0, device='cuda', opt_params={'lr':2e-3, 'iters':2500}, verbose=False): 
	W_f             = tensor.to(device) 
	params          = {}
	params['scale'] = torch.nn.Parameter(scale.float().to(device), requires_grad=True)
	params['zero']  = torch.nn.Parameter(zero.float().to(device),  requires_grad=True)
	optimizer       = torch.optim.AdamW([params[k] for k in params], lr=opt_params['lr'], betas=(0.9, 0.99), eps=1e-06, weight_decay=0.)

	def _loss_fct(output, target):
		return torch.mean(torch.abs(target - output)) #L1

	def _fake_quant():
		#Quantize
		W_q = torch.round(W_f*params['scale'] + params['zero']).clamp(min_max[0], min_max[1])
		#Dequantize
		W_r = (W_q - params['zero'])/params['scale'] 
		return W_r

	with torch.no_grad():
		_init_loss = _loss_fct(_fake_quant(), W_f).item()

	def _step():
		optimizer.zero_grad()
		loss    = _loss_fct(_fake_quant(), W_f)
		loss.backward()
		optimizer.step()
		return  np.round(loss.item(), 10)

	for i in range(opt_params['iters']):
		l = _step()
		if(verbose and (i%100)==0): print(i, l)
		
	with torch.no_grad():
		_final_loss = _loss_fct(_fake_quant(), W_f).item()

	if(_final_loss<_init_loss):
		for k in params: params[k] = params[k].data.detach().to(tensor.device)
	else:
		if(verbose): print('optimization failed...')
		params = {'scale':scale, 'zero':zero}

	del W_f	
	torch.cuda.empty_cache()
	return params['scale'], params['zero']

def is_divisible(val1, val2):
	return int(val2*np.ceil(val1/val2))==val1

def make_multiple(val, multiple):
	return int(multiple*np.ceil(val/float(multiple)))

def zero_pad_row(tensor, num_rows, dtype=None):
	out = torch.zeros([num_rows, tensor.shape[1]], device=tensor.device, dtype=tensor.dtype if (dtype is None) else dtype)
	out[:len(tensor)] = tensor
	return W_q

#Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
	@staticmethod
	def pack_8bit_u8(W_q):
		return W_q.to(torch.uint8)

	@staticmethod
	def unpack_8bit_u8(W_q):
		return W_q

	@staticmethod
	def pack_4bit_u8(W_q): #uint8 > uint8/2
		W_q = W_q.to(torch.uint8)
		_step = int(len(W_q)/2)
		return (W_q[:_step] << 4) | W_q[_step:]	

	@staticmethod
	def unpack_4bit_u8(W_q): #uint8/2 > uint8
		return torch.cat([(W_q & 0b11110000) >> 4, W_q & 0b00001111], axis=0)

	@staticmethod
	def pack_2bit_u8(W_q): #uint8 > uint8/4
		W_q = W_q.to(torch.uint8)
		_step = int(len(W_q)/4)
		return (W_q[:_step] << 6 | W_q[_step:2*_step] << 4 |  W_q[2*_step:3*_step] << 2 | W_q[3*_step:] )

	@staticmethod
	def unpack_2bit_u8(W_q):
		return torch.cat([(W_q & 0b11000000) >> 6, (W_q & 0b00110000) >> 4, (W_q & 0b00001100) >> 2, W_q & 0b00000011], axis=0)

	#int32 bit packing
	###################
	@staticmethod
	def pack_3bit_32(W_q_in):
		W_q = torch.zeros([int(10*np.ceil(W_q_in.shape[0]/10.)), W_q_in.shape[1]], device=W_q_in.device, dtype=torch.int32)
		W_q[:len(W_q_in)] = W_q_in
		_step = int(len(W_q)/10)
		W_q = (W_q[:_step] << 27) | (W_q[_step:_step*2] << 24) | (W_q[_step*2:_step*3] << 21) | (W_q[_step*3:_step*4] << 18) | (W_q[_step*4:_step*5] << 15) | (W_q[_step*5:_step*6] << 12) | (W_q[_step*6:_step*7] << 9) | (W_q[7*_step:_step*8] << 6) | (W_q[_step*8:_step*9] << 3) | (W_q[_step*9:]) 
		return W_q

	@staticmethod
	def unpack_3bit_32(W_q):
		return torch.cat([((W_q & 0b00111000000000000000000000000000) >> 27),
						  ((W_q & 0b00000111000000000000000000000000) >> 24),
						  ((W_q & 0b00000000111000000000000000000000) >> 21),
						  ((W_q & 0b00000000000111000000000000000000) >> 18),
						  ((W_q & 0b00000000000000111000000000000000) >> 15),
						  ((W_q & 0b00000000000000000111000000000000) >> 12),
						  ((W_q & 0b00000000000000000000111000000000) >> 9),
						  ((W_q & 0b00000000000000000000000111000000) >> 6),
						  ((W_q & 0b00000000000000000000000000111000) >> 3),
						  ((W_q & 0b00000000000000000000000000000111))], axis=0)

	@staticmethod
	def pack_3bit2bit_u8(W_q):
		assert is_divisible(len(W_q),3), "Input should have shape[0] divisble by 3 to use mixed 3-2bit bit packing"
		_step = int(len(W_q)/3)
		return (W_q[:_step] << 6 | W_q[1*_step:2*_step] << 3 | W_q[2*_step:] )

	@staticmethod
	def unpack_3bit2bit_u8(W_q):
		return torch.cat([(W_q & 0b11100000) >> 6, (W_q & 0b00011100) >> 3, W_q & 0b00000011], axis=0)

	@staticmethod
	def pack_4bit_32(W_q):
		W_q = W_q.to(torch.int32)
		_step = int(len(W_q)/8)
		W_q = (W_q[:_step] << 28) | (W_q[_step:_step*2] << 24) | (W_q[_step*2:_step*3] << 20) | (W_q[_step*3:_step*4] << 16) | (W_q[_step*4:_step*5] << 12) | (W_q[_step*5:_step*6] << 8) | (W_q[_step*6:_step*7] << 4) | (W_q[_step*7:])
		return W_q

	@staticmethod
	def unpack_4bit_32(W_q):
		return torch.cat([((W_q & 0b11110000000000000000000000000000) >> 28),
						  ((W_q & 0b00001111000000000000000000000000) >> 24),
						  ((W_q & 0b00000000111100000000000000000000) >> 20),
						  ((W_q & 0b00000000000011110000000000000000) >> 16),
						  ((W_q & 0b00000000000000001111000000000000) >> 12),
						  ((W_q & 0b00000000000000000000111100000000) >> 8),
						  ((W_q & 0b00000000000000000000000011110000) >> 4),
						  ((W_q & 0b00000000000000000000000000001111))], axis=0)



#Main HQQ Quantizer 
class Quantizer:
	SUPPORTED_BITS   = [8, 4, 3, 2]
	optimize_weights = optimize_weights_proximal

	bit_to_packing   = {8:'8bit_u8', 4:'4bit_u8', 3:'3bit_32', 2:'2bit_u8'}

	pack  =  {'8bit_u8':BitPack.pack_8bit_u8,
			  '4bit_u8':BitPack.pack_4bit_u8,
			  '3bit_32':BitPack.pack_3bit_32,
			  '2bit_u8':BitPack.pack_2bit_u8}

	unpack = {'8bit_u8':BitPack.unpack_8bit_u8,
			  '4bit_u8':BitPack.unpack_4bit_u8,
			  '3bit_32':BitPack.unpack_3bit_32,
			  '2bit_u8':BitPack.unpack_2bit_u8}

	@classmethod
	def quantize(cls, tensor, nbits=4, channel_wise=True, group_size=64, optimize=False, round_zero=False, axis=0):
		assert nbits in Quantizer.SUPPORTED_BITS, "nbits=" + str(nbits) + " not supported."
		assert axis in [0, 1], "axis should be either 0 or 1"
		if(group_size is not None):
			assert is_divisible(tensor.numel(), group_size), "group_size should be divisble by the total tensor dimensions. shape: " + str(tensor.shape) + ", group_size: " + str(group_size)

		W     = tensor.float() 
		shape = W.shape

		#Reshape for grouping
		if((group_size is not None) and channel_wise):
			W = W.reshape([-1, group_size]) if (axis==1) else W.reshape([group_size, -1])
		
		#Get min/max values
		if(channel_wise==False):
			_min, _max = W.min(), W.max()
			optimize   = False
		else:
			_min  = W.min(axis=axis, keepdim=True)[0]
			_max  = W.max(axis=axis, keepdim=True)[0]

		max_v   = 2**nbits - 1 
		min_v   = 0
		min_max = [min_v, max_v]

		#Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
		scale   = (max_v/(_max - _min)).clamp(max=2e4) #clamp to avoid half-precision problems
		zero    = -_min*scale 

		#Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
		if(round_zero): zero = torch.round(zero)
		
		#Fine-tune weights
		if(optimize): scale, zero = Quantizer.optimize_weights(tensor=W, scale=scale, zero=zero, min_max=min_max, axis=axis)

		#Quantize
		W_q  = torch.round(W*scale + zero).clamp(min_max[0], min_max[1])

		#Store meta-data (we invert the scale for dequantization)
		meta = {'nbits':nbits, 'group_size':group_size, 'shape':shape, 'scale':1./scale, 'zero':zero, 'axis':axis, 'packing':Quantizer.bit_to_packing[nbits]}

		#Pack bits
		W_q  = Quantizer.pack[meta['packing']](W_q)

		#cleanup
		del W, _min, _max 
		torch.cuda.empty_cache()

		return W_q, meta

	#Main dequantization: bit_unpacking > (W_q - z)*s > reshape
	@classmethod
	def dequantize(cls, W_q, meta):
		W_q_p = Quantizer.unpack[meta['packing']](W_q).half()
		if((meta['group_size'] is not None) and (meta['nbits']==3)):
			W_q_p = W_q_p[:meta['group_size']] if (meta['axis']==0) else W_q_p[:,:meta['group_size']]
		W_r = ((W_q_p - meta['zero'])*meta['scale']).reshape(meta['shape']) 
		del W_q_p
		return W_r

	@classmethod
	def to_inplace(cls, W_q, meta, device):
		W_q = W_q.to(device).contiguous() 
		for key in meta:
			if(type(meta[key])==torch.Tensor):
				meta[key] = (meta[key].half() if meta[key].dtype==torch.float32 else meta[key]).to(device).contiguous() 
		return W_q, meta

	@classmethod
	def to_ooplace(cls, W_q, meta, device):
		W_q_c  = W_q.to(device).contiguous() 
		meta_c = {}
		for key in meta:
			if(type(meta[key])==torch.Tensor):
				meta_c[key] = (meta[key].half() if meta[key].dtype==torch.float32 else meta[key]).to(device).contiguous() 
			else:
				meta_c[key] = meta[key]
		return W_q_c, meta_c

	@classmethod
	def cuda(cls, W_q, meta):
		return Quantizer.to_inplace(W_q, meta, device='cuda')

	@classmethod
	def cpu(cls, W_q, meta):
		return Quantizer.to_ooplace(W_q, meta, device='cpu')


#Main linear layer 
try:
	import hqq_aten
except:
	print('hqq_aten package not not installed.')
	hqq_aten = None

from enum import Enum
class HQQBackend(Enum):
	#Name of the forward functions
	PYTORCH         = "forward_pytorch" 
	PYTORCH_COMPILE = "forward_pytorch_compile"
	ATEN            = "forward_aten"

#Main linear layer 
class HQQLinear(torch.nn.Module):
	backend = HQQBackend.PYTORCH

	def __init__(self, linear_layer, quant_config, del_orig=True):
		super().__init__()
		self.ready        = False
		self.in_gpu       = False
		self.quant_config = quant_config
		if(linear_layer is not None):
			self.quantize(linear_layer.weight.data, **quant_config)
			self.bias = None if (linear_layer.bias==None) else linear_layer.bias.half().cuda()
		if(del_orig): del linear_layer
		torch.cuda.empty_cache()
		
	def cuda(self):
		if(self.in_gpu): return 
		self.W_q, self.meta = Quantizer.cuda(self.W_q, self.meta)
		if(self.meta['quant_scale']):
			self.meta['scale_q'] , self.meta['meta_scale'] = Quantizer.cuda(self.meta['scale_q'], self.meta['meta_scale'])
		if(self.meta['quant_zero']):
			self.meta['zero_q'] , self.meta['meta_zero']   = Quantizer.cuda(self.meta['zero_q'], self.meta['meta_zero'])
		self.in_gpu = True

	def to(self, device):
		pass

	def half(self):
		return self 

	def state_dict(self):
		return {'W_q':self.W_q, 'meta':self.meta, 'bias':self.bias}

	def load_state_dict(self, state_dict):
		self.W_q    = state_dict['W_q']
		self.meta   = state_dict['meta']
		self.bias   = state_dict['bias'] if ('bias' in state_dict) else None
		self.in_gpu = self.W_q.device.type == 'cuda'
		if(self.in_gpu==False): self.cuda()
		self.ready  = True

	def quantize(self, W, weight_quant_params, scale_quant_params, zero_quant_params):
		quant_scale = scale_quant_params is not None
		quant_zero  = zero_quant_params  is not None
		
		#Quantize
		W_q , meta = Quantizer.quantize(W, **weight_quant_params) 
		meta.update({'quant_scale':quant_scale, 'quant_zero':quant_zero})
		if(meta['quant_scale']):
			meta['scale_q'] , meta['meta_scale'] = Quantizer.quantize(meta['scale'], **scale_quant_params); del meta['scale']
		if(meta['quant_zero']):
			meta['zero_q'], meta['meta_zero']    = Quantizer.quantize(meta['zero'],  **zero_quant_params);  del meta['zero']

		self.W_q   = W_q
		self.meta  = meta 
		self.cuda()
		self.ready = True

	@torch.inference_mode()
	def dequantize(self):
		assert self.ready, "model was not quantized"
		W_q, meta = self.W_q, self.meta
		del_keys = []
		if(meta['quant_scale']):
			meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
		if(meta['quant_zero']):
			meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')
		W_est = Quantizer.dequantize(W_q, meta)
		#Cleanup
		for key in del_keys: del meta[key]
		return W_est

	def forward(self, x):
		return getattr(self, HQQLinear.backend.value)(x)

	@torch.no_grad()
	def forward_pytorch(self, x):
		W_est = self.dequantize()
		out   = torch.matmul(x, W_est.t())
		if(self.bias!=None): out += self.bias
		del W_est
		return out

	##############################################
	#Experimental
	#############################################
	@torch.no_grad()
	@torch.compile()
	def forward_pytorch_compile(self, x):
		W_est = self.dequantize()
		out   = torch.matmul(x, W_est.t())
		if(self.bias!=None): out += self.bias
		del W_est
		return out

	@torch.no_grad()
	def forward_aten(self, x):
		empt = torch.empty([0])
		W_q  = self.W_q
		meta = self.meta
		bias = self.bias

		W_q, W_s, W_z              = W_q,  empt if (meta['quant_scale']) else meta['scale'], empt if (meta['quant_zero']) else meta['zero']
		W_shape,  W_group_size     = meta['shape'], meta['group_size']
		W_nbits, W_axis, W_packing = meta['nbits'], meta['axis'], meta['packing']

		if(meta['quant_scale']):
			S_q, S_s, S_z              = meta['scale_q'],             meta['meta_scale']['scale'], meta['meta_scale']['zero']
			S_shape, S_group_size      = meta['meta_scale']['shape'], meta['meta_scale']['group_size'] 
			S_nbits, S_axis, S_packing = meta['meta_scale']['nbits'], meta['meta_scale']['axis'],  meta['meta_scale']['packing']
		else:
			S_q, S_s, S_z              = empt, empt, empt
			S_shape, S_group_size      = meta['shape'], -1
			S_nbits, S_axis, S_packing = -1, 0, ""

		if(meta['quant_zero']):
			Z_q, Z_s, Z_z              = meta['zero_q'],             meta['meta_zero']['scale'], meta['meta_zero']['zero']
			Z_shape, Z_group_size      = meta['meta_zero']['shape'], meta['meta_zero']['group_size']
			Z_nbits, Z_axis, Z_packing = meta['meta_zero']['nbits'], meta['meta_zero']['axis'],  meta['meta_zero']['packing']
		else:
			S_q, S_s, S_z              = empt, empt, empt
			S_shape, S_group_size      = meta['shape'], -1
			S_nbits, S_axis, S_packing = -1, 0, ""


		S_group_size = 0 if (S_group_size==None) else S_group_size
		Z_group_size = 0 if (Z_group_size==None) else Z_group_size

		args = [x, bias if (bias is not None) else empt,
				W_q, W_s, W_z, W_shape, W_group_size, W_nbits, W_axis, W_packing,
				S_q, S_s, S_z, S_shape, S_group_size, S_nbits, S_axis, S_packing,
				Z_q, Z_s, Z_z, Z_shape, Z_group_size, Z_nbits, Z_axis, Z_packing]

		return hqq_aten.forward_with_quant(*args)



def hqq_base_quant_config(nbits=4, group_size=64, quant_zero=True, quant_scale=False):
	assert nbits in Quantizer.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
	assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8." 
	weight_quant_params = {'nbits':nbits,'channel_wise':True,  'group_size':group_size, 'optimize':True, 'round_zero':True if nbits==4 else False} 
	scale_quant_params  = {'nbits':8,    'channel_wise':True,  'group_size':128,        'optimize':False} if (quant_scale) else None
	zero_quant_params   = {'nbits':8,    'channel_wise':False, 'group_size':None,       'optimize':False} if (quant_zero)  else None
	return {'weight_quant_params':weight_quant_params, 'scale_quant_params':scale_quant_params, 'zero_quant_params':zero_quant_params}
