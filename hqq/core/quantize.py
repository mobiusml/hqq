#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch, copy
import numpy as np 

from .utils    import *
from .optimize import *
from .bitpack  import BitPack 
from tqdm import tqdm
from termcolor import colored

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
 
	unpack_view_dtype = {'8bit_u8':torch.uint8,
						 '4bit_u8':torch.uint8,
						 '3bit_32':torch.int32,
						 '2bit_u8':torch.uint8}

	@classmethod
	def quantize(cls, tensor, nbits=4, channel_wise=True, group_size=64, optimize=False, round_zero=False, axis=0, bitpack=True):
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
		scale, zero = scale.clone(), zero.clone() #Necessary for fake quantization backprop
		W_q  = torch.round(W*scale + zero).clamp(min_max[0], min_max[1])

		#Store meta-data (we invert the scale for dequantization)
		meta = {'nbits':nbits, 'group_size':group_size, 'shape':shape, 'scale':1./scale, 'zero':zero, 'axis':axis, 'packing':Quantizer.bit_to_packing[nbits]}
		meta['unpack_view_dtype'] = Quantizer.unpack_view_dtype[meta['packing']]
  
		#Pack bits
		if(bitpack):
			W_q = Quantizer.pack[meta['packing']](W_q).view(torch.float32) # store quantized weights as float32
		else:
			W_q = W_q.to(tensor.dtype) 
			meta['packing'] = None

		#cleanup
		del W, _min, _max 
		torch.cuda.empty_cache()

		return W_q, meta

	#Main dequantization: bit_unpacking > (W_q - z)*s > reshape
	@classmethod
	def dequantize(cls, W_q, meta):
		compute_dtype = meta['compute_dtype'] if ('compute_dtype' in meta) else torch.float16
		if(meta['packing']):
			W_r = Quantizer.unpack[meta['packing']](W_q.view(meta['unpack_view_dtype'])).to(compute_dtype)
			if((meta['group_size'] is not None) and (meta['nbits']==3)):
				W_r = W_r[:meta['group_size']] if (meta['axis']==0) else W_r[:,:meta['group_size']]
		else:
			W_r = W_q.to(compute_dtype)
		W_r = ((W_r - meta['zero'])*meta['scale']).reshape(meta['shape']) 
		return W_r

	@classmethod
	def to_inplace(cls, W_q, meta, device):
		compute_dtype = meta['compute_dtype'] if ('compute_dtype' in meta) else torch.float16
		if(W_q is not None):
			W_q = W_q.to(device).contiguous() 
		for key in meta:
			if(type(meta[key])==torch.Tensor):
				meta[key] = (meta[key].to(compute_dtype) if torch.is_floating_point(meta[key]) else meta[key]).to(device).contiguous() 
		return W_q, meta

	@classmethod
	def to_ooplace(cls, W_q, meta, device):
		compute_dtype = meta['compute_dtype'] if ('compute_dtype' in meta) else torch.float16
		if(W_q is not None):
			W_q_c  = W_q.to(device).contiguous() 
		else:
			W_q_c = None 
		meta_c = {}
		for key in meta:
			if(type(meta[key])==torch.Tensor):
				meta_c[key] = (meta[key].to(compute_dtype) if torch.is_floating_point(meta[key]) else meta[key]).to(device).contiguous() 
			else:
				meta_c[key] = meta[key]
		return W_q_c, meta_c

	@classmethod
	def cuda(cls, W_q, meta, device_n=0):
		return Quantizer.to_inplace(W_q, meta, device='cuda:' + str(device_n))

	@classmethod
	def cpu(cls, W_q, meta):
		return Quantizer.to_ooplace(W_q, meta, device='cpu')


#Main linear layer 
try:
	import hqq_aten
	print(colored('hqq_aten package available. Set backend to HQQBackend.ATEN for faster inference and HQQBackend.ATEN_BACKPROP for faster training!', 'cyan'))
except:
	hqq_aten = None

from enum import Enum
class HQQBackend(Enum):
	#Name of the forward functions
	PYTORCH                  = "forward_pytorch" 
	PYTORCH_COMPILE          = "forward_pytorch_compile"
	PYTORCH_BACKPROP         = "forward_pytorch_backprop" 
	PYTORCH_BACKPROP_COMPILE = "forward_pytorch_backprop_compile" 
	ATEN                     = "forward_aten"
	ATEN_BACKPROP            = "forward_aten_backprop"


#No cache: less memory, slower
class HQQMatmulNoCacheDeq(torch.autograd.Function):

	@staticmethod
	def forward(x, dequantize, bias):
		out = torch.matmul(x, dequantize().t())
		if(bias!=None): out += bias
		return out

	@staticmethod
	def setup_context(ctx, inputs, outputs):
		x, dequantize, bias = inputs
		ctx.save_for_backward(x, bias)
		ctx.dequantize = dequantize

	@staticmethod
	def backward(ctx, grad_output):
		x, bias    = ctx.saved_tensors
		dtype_out  = grad_output.dtype

		grad_input = grad_weight = grad_bias = None

		if ctx.needs_input_grad[0]:
			grad_input = torch.matmul(grad_output, ctx.dequantize()) 

		# weight grad for frozen quantized weights not defined
		# if ctx.needs_input_grad[1]:
		# 	grad_weight = torch.matmul(grad_output.t(), x)

		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)

		return grad_input, grad_weight, grad_bias


class HQQMatmulNoCacheMul(torch.autograd.Function):

	@staticmethod
	def forward(x, matmul, bias):
		out = matmul(x, transpose=True)
		if(bias!=None): out += bias
		return out

	@staticmethod
	def setup_context(ctx, inputs, outputs):
		x, matmul, bias = inputs
		ctx.save_for_backward(x, bias)
		ctx.matmul = matmul

	@staticmethod
	def backward(ctx, grad_output):
		x, bias    = ctx.saved_tensors

		grad_input = grad_weight = grad_bias = None

		if ctx.needs_input_grad[0]:
			grad_input = ctx.matmul(grad_output, transpose=False)

		# weight grad for frozen quantized weights not defined
		# if ctx.needs_input_grad[1]:
		# 	grad_weight = torch.matmul(grad_output.t(), x)

		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)

		return grad_input, grad_weight, grad_bias

#Cache dequantized tensor: Faster but needs more memory
class HQQMatmulCachedDeq(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, hqq_layer, bias):
		weight_tmp = hqq_layer.dequantize() 
		out        = torch.matmul(x, weight_tmp.t())
		if(bias!=None): out += bias

		ctx.save_for_backward(x, bias, weight_tmp)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		x, bias, weight_tmp = ctx.saved_tensors

		grad_input = grad_weight = grad_bias = None

		if ctx.needs_input_grad[0]:
			grad_input = torch.matmul(grad_output, weight_tmp) 

		del weight_tmp

		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)

		return grad_input, grad_weight, grad_bias

#Main linear layer 
class HQQLinear(torch.nn.Module):
	backend = HQQBackend.PYTORCH #Default

	def __init__(self, linear_layer, quant_config, del_orig=True, compute_dtype=torch.float16, device_n=0, initialize=True):
		super().__init__()
		self.ready         = False
		self.in_gpu        = False
		self.device        = None
		self.bias          = None
		self.device_n      = device_n
		self.compute_dtype = compute_dtype
		self.quant_config  = copy.deepcopy(quant_config)
		self.del_orig = del_orig
		self.offload_meta  = self.quant_config.pop('offload_meta') if (self.quant_config is not None) else None
		
		self.set_backend(HQQLinear.backend) #Default backend

		self.linear_layer = linear_layer
		self.bias = None if (self.linear_layer.bias==None) else self.linear_layer.bias.to(self.compute_dtype).cuda()
		if(initialize): self.initialize()
  
	def initialize(self):
		if(self.linear_layer is not None):
			self.quantize(self.linear_layer.weight.data, **self.quant_config)

		if(self.del_orig): del self.linear_layer
		torch.cuda.empty_cache()
  
	#Set backends
	@classmethod
	def set_backend(cls, backend: HQQBackend):
		HQQLinear.backend = backend
		cls.forward       = getattr(cls, backend.value)

	#TODO: rewrite this mess 
	def cuda(self, device_n=0):
		if(self.in_gpu): return 
		self.meta['compute_dtype'] = self.compute_dtype 

		self.W_q, self.meta = Quantizer.cuda(self.W_q, self.meta, device_n)

		if(self.meta['quant_zero']):
			self.meta['zero_q'] , self.meta['meta_zero']   = Quantizer.cuda(self.meta['zero_q'], self.meta['meta_zero'], device_n)

		if(self.meta['quant_scale']):
			self.meta['scale_q'] , self.meta['meta_scale'] = Quantizer.cuda(self.meta['scale_q'], self.meta['meta_scale'], device_n)
		
		if(self.offload_meta):
			if(self.meta['quant_scale'] and self.meta['quant_zero']):
				self.meta['zero_scale'] = torch.stack((self.meta['zero_q'], self.meta['scale_q']))
				del self.meta['scale_q'], self.meta['zero_q']
			else:
				self.meta['zero_scale'] = torch.stack((self.meta['zero'], self.meta['scale'])).to(self.compute_dtype)
				del self.meta['scale'], self.meta['zero'] 

			self.meta['zero_scale'] = self.meta['zero_scale'].contiguous().cpu()

		if(self.bias is not None):
			self.bias = self.bias.to(self.compute_dtype).cuda(device_n)

		self.W_q    = torch.nn.Parameter(self.W_q, requires_grad=False)
		self.device = self.W_q.device
		self.in_gpu = True

		torch.cuda.empty_cache()

	def to(self, *args, **kwargs):
		pass

	def half(self, *args, **kwargs):
		return self 

	def state_dict(self):
		return {'W_q':self.W_q, 'meta':self.meta, 'bias':self.bias}

	def load_state_dict(self, state_dict):
		self.W_q    = state_dict['W_q']
		self.meta   = state_dict['meta']
		self.bias   = state_dict['bias'] if ('bias' in state_dict) else None
		self.in_gpu = self.W_q.device.type == 'cuda'
		if(self.in_gpu): 
			if('zero' in self.meta):
				self.meta['zero']  = self.meta['zero'].to(self.compute_dtype)

			if('scale' in self.meta):
				self.meta['scale'] = self.meta['scale'].to(self.compute_dtype)

			if(('zero_scale' in self.meta) and (self.meta['quant_scale']==False) and (self.meta['zero_scale']==False)):
				self.meta['zero_scale']  = self.meta['zero_scale'].to(self.compute_dtype)
		else:
			self.cuda(self.device_n)
		self.ready  = True

	def quantize(self, W, weight_quant_params, scale_quant_params, zero_quant_params):
		quant_scale = scale_quant_params is not None
		quant_zero  = zero_quant_params  is not None

		self.in_features, self.out_features = W.t().shape
		
		#Quantize
		W_q , meta = Quantizer.quantize(W, **weight_quant_params) 
		meta.update({'quant_scale':quant_scale, 'quant_zero':quant_zero})

		if(meta['quant_zero']):
			meta['zero_q'], meta['meta_zero']    = Quantizer.quantize(meta['zero'],  **zero_quant_params);  del meta['zero']
			meta['meta_zero']['compute_dtype']   = self.compute_dtype

		if(meta['quant_scale']):
			meta['scale_q'] , meta['meta_scale'] = Quantizer.quantize(meta['scale'], **scale_quant_params); del meta['scale']
			meta['meta_scale']['compute_dtype']  = self.compute_dtype

		self.W_q   = W_q
		self.meta  = meta 
		self.cuda(self.device_n)
		self.ready = True

	def dequantize(self):
		assert self.ready, "model was not quantized"
		W_q, meta = self.W_q, self.meta

		del_keys = []

		if('zero_scale' in meta):
			zero_scale = meta['zero_scale'].to(self.W_q.device)

			if(zero_scale.dtype==torch.uint8):
				meta['zero_q']  = zero_scale[0]; del_keys.append('zero_q');
				meta['scale_q'] = zero_scale[1]; del_keys.append('scale_q');
			else:
				meta['zero']  = zero_scale[0]; del_keys.append('zero');
				meta['scale'] = zero_scale[1]; del_keys.append('scale');

		if(meta['quant_zero']):
			meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')

		if(meta['quant_scale']):
			meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')

		W_est = Quantizer.dequantize(W_q, meta)

		#Cleanup
		for key in del_keys: del meta[key]
		return W_est

	def matmul(self, x, transpose=True):
		weight = self.dequantize() 
		return torch.matmul(x, weight.t() if (transpose) else weight)

	@torch.compile()
	def matmul_compile(self, *args, **kwargs):
		return self.matmul(*args, **kwargs)

	def forward_pytorch_backprop(self, x):
		return HQQMatmulNoCacheMul.apply(x, self.matmul, self.bias)

	def forward_pytorch_backprop_compile(self, x):
		return HQQMatmulNoCacheMul.apply(x, self.matmul_compile, self.bias)

	def forward_pytorch(self, x): 
		out = torch.matmul(x, self.dequantize().t())
		if(self.bias is not None):
			out += self.bias
		return out

	@torch.compile()
	def forward_pytorch_compile(self, x): 
		return self.forward_pytorch(x)

	##############################################
	#Experimental
	#############################################
	#Requires building the aten backend
	@torch.jit.ignore
	def dequantize_Wq_aten(self, W_q, meta):
		return hqq_aten.dequantize(W_q, meta['scale'], meta['zero'], meta['shape'], meta['group_size'] if (meta['group_size']) else -1, meta['nbits'], meta['axis'], meta['packing'])

	def dequantize_aten(self):
		#Dequantize 
		assert self.ready, "model was not quantized"
		W_q, meta = self.W_q, self.meta

		del_keys = []

		if('zero_scale' in meta):
			zero_scale = meta['zero_scale'].to(self.W_q.device)

			if(zero_scale.dtype==torch.uint8):
				meta['zero_q']  = zero_scale[0]; del_keys.append('zero_q');
				meta['scale_q'] = zero_scale[1]; del_keys.append('scale_q');
			else:
				meta['zero']  = zero_scale[0]; del_keys.append('zero');
				meta['scale'] = zero_scale[1]; del_keys.append('scale');

		if(meta['quant_scale']):
			if(meta['meta_scale']['group_size']):
				meta['scale'] = self.dequantize_Wq_aten(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
			else:
				meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')

		if(meta['quant_zero']):
			if(meta['meta_zero']['group_size']):
				meta['zero'] = self.dequantize_Wq_aten(meta['zero_q'], meta['meta_zero']); del_keys.append('zero')
			else:
				meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')
		
		# print("Dtype and shape:", W_q.shape, W_q.dtype)
		
		if meta['packing']: W_q = W_q.view(meta['unpack_view_dtype'])

		# print("Dtype and shape:", W_q.shape, W_q.dtype)
  
		W_est = self.dequantize_Wq_aten(W_q, meta)

		#Cleanup
		for key in del_keys: del meta[key]

		return W_est

	def forward_aten(self, x):
		W_est = self.dequantize_aten()
		out   = torch.matmul(x, W_est.t())
		if(self.bias is not None):
			out += self.bias
		return out

	def forward_aten_backprop(self, x):
		return HQQMatmulNoCacheDeq.apply(x, self.dequantize_aten, self.bias)


	# def forward_aten(self, x):
	# 	empt = torch.empty([0])
	# 	W_q  = self.W_q
	# 	meta = self.meta
	# 	bias = self.bias

	# 	W_q, W_s, W_z              = W_q,  empt if (meta['quant_scale']) else meta['scale'], empt if (meta['quant_zero']) else meta['zero']
	# 	W_shape,  W_group_size     = meta['shape'], meta['group_size']
	# 	W_nbits, W_axis, W_packing = meta['nbits'], meta['axis'], meta['packing']

	# 	if(meta['quant_scale']):
	# 		S_q, S_s, S_z              = meta['scale_q'],             meta['meta_scale']['scale'], meta['meta_scale']['zero']
	# 		S_shape, S_group_size      = meta['meta_scale']['shape'], meta['meta_scale']['group_size'] 
	# 		S_nbits, S_axis, S_packing = meta['meta_scale']['nbits'], meta['meta_scale']['axis'],  meta['meta_scale']['packing']
	# 	else:
	# 		S_q, S_s, S_z              = empt, empt, empt
	# 		S_shape, S_group_size      = meta['shape'], -1
	# 		S_nbits, S_axis, S_packing = -1, 0, ""

	# 	if(meta['quant_zero']):
	# 		Z_q, Z_s, Z_z              = meta['zero_q'],             meta['meta_zero']['scale'], meta['meta_zero']['zero']
	# 		Z_shape, Z_group_size      = meta['meta_zero']['shape'], meta['meta_zero']['group_size']
	# 		Z_nbits, Z_axis, Z_packing = meta['meta_zero']['nbits'], meta['meta_zero']['axis'],  meta['meta_zero']['packing']
	# 	else:
	# 		S_q, S_s, S_z              = empt, empt, empt
	# 		S_shape, S_group_size      = meta['shape'], -1
	# 		S_nbits, S_axis, S_packing = -1, 0, ""


	# 	S_group_size = 0 if (S_group_size==None) else S_group_size
	# 	Z_group_size = 0 if (Z_group_size==None) else Z_group_size

	# 	args = [x, bias if (bias is not None) else empt,
	# 			W_q, W_s, W_z, W_shape, W_group_size, W_nbits, W_axis, W_packing,
	# 			S_q, S_s, S_z, S_shape, S_group_size, S_nbits, S_axis, S_packing,
	# 			Z_q, Z_s, Z_z, Z_shape, Z_group_size, Z_nbits, Z_axis, Z_packing]

	# 	return hqq_aten.forward_with_quant(*args)


def hqq_base_quant_config(nbits=4, group_size=64, quant_zero=True, quant_scale=False, offload_meta=False):
	assert nbits in Quantizer.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
	if(group_size is not None):
		assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8." 
	weight_quant_params = {'nbits':nbits,'channel_wise':True,  'group_size':group_size, 'optimize':True, 'round_zero':True if nbits==4 else False} 

	if(offload_meta):
		if((quant_scale!=quant_zero)):
			print(colored("quant_zero and quant_scale must be the same when offload_meta is set to True. Setting quant_scale=quant_zero." , 'yellow'))
			quant_scale = quant_zero

		scale_quant_params  = {'nbits':8, 'channel_wise':True,  'group_size':128,  'optimize':False} if (quant_scale) else None
		zero_quant_params   = {'nbits':8, 'channel_wise':True,  'group_size':128,  'optimize':False} if (quant_zero)  else None

	else:
		scale_quant_params  = {'nbits':8, 'channel_wise':True,  'group_size':128,  'optimize':False} if (quant_scale) else None
		zero_quant_params   = {'nbits':8, 'channel_wise':False, 'group_size':None, 'optimize':False} if (quant_zero)  else None


	return {'weight_quant_params':weight_quant_params, 'scale_quant_params':scale_quant_params, 'zero_quant_params':zero_quant_params, 'offload_meta':offload_meta}

#Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = hqq_base_quant_config