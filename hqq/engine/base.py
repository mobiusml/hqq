from abc import abstractmethod
from typing import Dict
from ..models.base import BaseHQQModel

#Wrapper that makes it easier to add quantization support to different engines (HF, VLLM, etc.)

class HQQWrapper:

	@abstractmethod
	def _get_arch_key_from_save_dir(cls, save_dir:str):
		pass

	@classmethod
	def _get_hqq_class(cls, arg):
		arch = arg if (type(arg)==str) else arg.arch_key
		return cls._HQQ_REGISTRY[arch]

	@classmethod
	def _validate_params(cls, params:Dict):
		pass

	@classmethod
	def _is_quantizable(cls, model):
		return hasattr(model, 'hqq_quantized')

	@classmethod
	def _make_quantizable(cls, model, quantized):
		model.hqq_quantized = quantized

	@classmethod
	def _check_arch_support(cls, arg):
		arch = arg if (type(arg)==str) else arg.arch_key
		assert (arch in cls._HQQ_REGISTRY), "Model architecture " + arch + " not supported yet."

	@classmethod
	def _check_if_already_quantized(cls, model):
		assert (not model.hqq_quantized), "Model already quantized"

	@classmethod
	def _check_if_not_quantized(cls, model):
		assert model.hqq_quantized, "Model not quantized."

	@classmethod
	def _set_quantized(cls, model, quantized):
		model.hqq_quantized = quantized

	#####################################################
	@classmethod
	def quantize_model_(cls, model, quant_config):
		if(cls._is_quantizable(model)==False):
			cls._make_quantizable(model, quantized=False)  
		cls._check_arch_support(model)
		cls._check_if_already_quantized(model)
		cls._get_hqq_class(model).quantize_model(model, quant_config=quant_config)
		cls._set_quantized(model, True)

	@classmethod
	def save_quantized_(cls, model, save_dir):
		cls._check_if_not_quantized(model)
		cls._get_hqq_class(model).save_quantized(model, save_dir=save_dir)

	@classmethod
	def from_quantized(cls, save_dir_or_hub, cache_dir=''):
		#Both local and hub-support
		save_dir = BaseHQQModel.try_snapshot_download(save_dir_or_hub)
		arch_key = cls._get_arch_key_from_save_dir(save_dir)
		cls._check_arch_support(arch_key)

		model = cls._get_hqq_class(arch_key).from_quantized(save_dir, cache_dir)

		cls._make_quantizable(model, quantized=True)
		return model

	@classmethod
	def get_linear_tags(cls, model):
		return cls._get_hqq_class(model).get_linear_tags()

