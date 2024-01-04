import torch 
import numpy as np 
from .quantize import HQQLinear, HQQBackend, Quantizer

def _get_dense_param(in_features, out_features, device='cuda', trainable=True, dtype=torch.bfloat16):
	W = torch.nn.Linear(in_features, out_features, bias=None).weight.data.t().to(dtype).to(device).contiguous()
	return torch.nn.Parameter(W, requires_grad=trainable)

class HQQLinearLoRA(torch.nn.Module):
	def __init__(self, linear_layer, peft_config):
		super().__init__()

		#Device
		self.device        = next(linear_layer.parameters()).device
		self.train_dtype   = peft_config['train_dtype'] if ('train_dtype' in peft_config) else torch.float

		#Linear layer
		self.linear_layer  = linear_layer
		self.in_features   = linear_layer.in_features
		self.out_features  = linear_layer.out_features
		self.bias          = None if (linear_layer.bias is None) else linear_layer.bias.clone()

		#Turn-off bias in the linear layer
		self.linear_layer.bias = None

		#Dropout
		if('dropout' in peft_config):
			self.peft_drop  = torch.nn.Dropout(p=peft_config['dropout']) if (peft_config['dropout']>0.) else torch.nn.Identity()
		else:
			self.peft_drop  = torch.nn.Identity()

		#LoRA A/B
		self.peft_config = peft_config
		self.lora_alpha  = peft_config['lora_alpha']
		self.r           = peft_config['r']
		self.scaling     = self.lora_alpha/self.r
		self.lora_A      = _get_dense_param(self.in_features, self.r,  device=self.device, trainable=True, dtype=self.train_dtype) 
		self.lora_B      = _get_dense_param(self.r, self.out_features, device=self.device, trainable=True, dtype=self.train_dtype) 

		#Init weights, as as the original LoRA implementation 
		torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
		torch.nn.init.zeros_(self.lora_B)

	def forward(self, x):
		x_type = x.dtype 

		#Forward with base linear 
		out = self.linear_layer(x)
		
		#LoRA
		out += (torch.matmul(torch.matmul(self.peft_drop(x.to(self.lora_A.dtype)), self.lora_A), self.lora_B)*self.scaling).to(x_type)

		#Bias
		if(self.bias is not None):
			out += self.bias
		
		return out

	def merge_and_quantize(self, quant_config):

		#Get initial weights
		W = self.linear_layer(torch.eye(self.in_features, device=self.device, dtype=torch.float16)).t()

		#Merge weights
		W += (torch.matmul(self.lora_A.data, self.lora_B.data).t()*self.scaling).to(W.dtype)

		new_hqq_layer = HQQLinear(None, quant_config)
		new_hqq_layer.bias = None if (self.bias is None) else self.bias.clone() 
		new_hqq_layer.quantize(W, **quant_config)

		return new_hqq_layer

	def cast(self, dtype=torch.float16):
		self.lora_A.data = self.lora_A.data.to(dtype)
		self.lora_B.data = self.lora_B.data.to(dtype)
		if(self.bias is not None):
			if(self.bias.requires_grad):
				self.bias.data = self.bias.data.to(dtype)
			else:
				self.bias = self.bias.to(dtype)
		return self

	def state_dict(self):
		return {'lora_A':self.lora_A.data, 'lora_B':self.lora_B.data, 'scaling':self.scaling, 'bias':self.bias, 'peft_config':self.peft_config}

	def load_state_dict(self, state_dict):
		self.lora_A.data = state_dict['lora_A'].data.to(self.device)
		self.lora_B.data = state_dict['lora_B'].data.to(self.device)
		self.scaling     = state_dict['scaling']
		self.bias        = state_dict['bias'] if ('bias' in state_dict) else None
		self.bias        = self.bias.to(self.device) if (self.bias is not None) else None
		self.peft_config = state_dict['peft_config']


#LoRA with fake quantization 
class HQQLinearLoRAWithFakeQuant(HQQLinearLoRA):
	def __init__(self, linear_layer, peft_config, quant_param):
		super(HQQLinearLoRAWithFakeQuant, self).__init__(linear_layer, peft_config)
		self.quant_param = quant_param

	def fake_quant(self, weight):
		if(self.quant_param):
			W_q, meta  = Quantizer.quantize(weight, **self.quant_param, bitpack=False) 
			weight_est = Quantizer.dequantize(W_q, meta)
		else:
			weight_est = weight
		return weight_est

	def forward(self, x):
		weight = self.linear_layer.dequantize() + (torch.matmul(self.lora_A, self.lora_B)*self.scaling).t()
		weight = self.fake_quant(weight)
		out    = torch.matmul(x, weight.t())
		#Bias
		if(self.bias is not None):
			out += self.bias

		return out 


_HQQ_LORA_CLASSES = [HQQLinearLoRA, HQQLinearLoRAWithFakeQuant]
_HQQ_LORA_MAPPING = {'default':HQQLinearLoRA, 'lora_with_fakequant':HQQLinearLoRAWithFakeQuant}

def is_hqq_lora_layer(layer):
	return type(layer) in _HQQ_LORA_CLASSES
##################################################################################################################
def autoname_modules(model):
	for name, module in model.named_modules():
		module.name = name

#Patching functions
def patch_linear_add_peft(layer, patch_params):
	_peft_config = patch_params
	if(_peft_config):
		lora_type = _peft_config['lora_type'] if ('lora_type' in _peft_config) else 'default'
		new_layer = _HQQ_LORA_MAPPING[lora_type](layer, _peft_config)
	else:
		new_layer = layer
	return new_layer

def patch_linear_merge_peft(layer, patch_params):
	_quant_config = patch_params
	if(_quant_config):
		new_layer = layer.merge_and_quantize(_quant_config)
		del layer
		cleanup()
	else:
		new_layer = layer
	return new_layer

def patch_linear_cast_peft(layer, patch_params):
	if(is_hqq_lora_layer(layer)):
		layer.cast(patch_params)
	return layer

#Putting it all together
class PeftUtils:

	@classmethod
	def get_base_class(cls, model, base_class):
		#Get base class
		if((base_class is None) and hasattr(model, 'base_class')):
			base_class = model.base_class

		assert (base_class is not None), "You need to provide the base HQQ class (LlamaHQQ, MixtralHQQ, etc.) as model.base_class or as an argument base_class=LlamaHQQ"
		return base_class
	
	@classmethod
	def add_lora(cls, model, lora_params, base_class=None, verbose=True):

		#Base classs
		base_class = cls.get_base_class(model, base_class)

		#Freeze
		for param in model.parameters():
			param.requires_grad = False

		#Patch
		base_class.patch_linearlayers(model, patch_linear_add_peft, lora_params, verbose=verbose)

		#Rename modules
		autoname_modules(model)

		#Default backprop backend
		HQQLinear.set_backend(HQQBackend.PYTORCH_BACKPROP)

	@classmethod
	def merge_lora(cls, model, merge_lora_params, base_class=None, verbose=True):
		#Base classs
		base_class = cls.get_base_class(model, base_class)

		#Patch
		base_class.patch_linearlayers(model, patch_linear_merge_peft, merge_lora_params, verbose=verbose)

	@classmethod
	def cast_lora_weights(cls, model, dtype, base_class=None, verbose=True):
		#Base classs
		base_class = cls.get_base_class(model, base_class)

		#Linear tags
		linear_tags = base_class.get_linear_tags()

		#Patch
		base_class.patch_linearlayers(model, 
									  patch_linear_cast_peft, 
									  dict([(linear_tag, dtype) for linear_tag in linear_tags]), 
									  verbose=verbose)


	@classmethod
	def save_lora_weights(cls, model, filename, base_class=None, verbose=True):
		#Base classs
		base_class = cls.get_base_class(model, base_class)

		lora_global_params = {}
		def _patch_linear_save_weights(layer, patch_params, return_layer=True):
			if(is_hqq_lora_layer(layer)):
				lora_global_params[layer.name] = layer.state_dict()
			if(return_layer): return layer

		#Linear tags
		linear_tags = base_class.get_linear_tags()

		#Patch
		base_class.patch_linearlayers(model, 
									  _patch_linear_save_weights, 
									  dict([(linear_tag, None) for linear_tag in linear_tags]), 
									  verbose=verbose)

		#save
		torch.save(lora_global_params, filename)

	@classmethod
	def load_lora_weights(cls, model, filename, base_class=None, verbose=True):
		#Base classs
		base_class = cls.get_base_class(model, base_class)

		lora_global_params = torch.load(file, map_location='cpu')

		def _patch_linear_load_weights(layer, patch_params, return_layer=True):
			if(is_hqq_lora_layer(layer)):
				layer.load_state_dict(lora_global_params[layer.name])
			if(return_layer): return layer

		#Linear tags
		linear_tags = base_class.get_linear_tags()

		#Patch
		base_class.patch_linearlayers(model, 
									  _patch_linear_load_weights, 
									  dict([(linear_tag, None) for linear_tag in linear_tags]), 
									  verbose=verbose)
