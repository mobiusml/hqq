import timm, json
from typing import Dict
from ..models.base import BaseHQQModel
from ..models.timm.vit_clip import ViTCLIPHQQ

_HQQ_REGISTRY = {}
_HQQ_REGISTRY['vit_huge_patch14_clip_336']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_huge_patch14_clip_224']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_large_patch14_clip_224'] = ViTCLIPHQQ
_HQQ_REGISTRY['vit_large_patch14_clip_336'] = ViTCLIPHQQ
_HQQ_REGISTRY['vit_base_patch16_clip_384']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_base_patch32_clip_448']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_base_patch32_clip_384']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_base_patch16_clip_224']  = ViTCLIPHQQ
_HQQ_REGISTRY['vit_base_patch32_clip_224']  = ViTCLIPHQQ

from .base import HQQWrapper
class HQQtimm(HQQWrapper):
	_HQQ_REGISTRY = _HQQ_REGISTRY

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		cls._make_quantizable(model, quantized=False)

	@classmethod
	def _make_quantizable(cls, model, quantized):
		model.hqq_quantized  = quantized
		model.arch_key       = model.default_cfg['architecture']
		model.quantize_model = lambda quant_config: cls.quantize_model_(model=model, quant_config=quant_config)
		model.save_quantized = lambda save_dir: cls.save_quantized_(model=model, save_dir=save_dir)
		model.cuda           = lambda *args, **kwargs: model if(quantized) else model.cuda
		model.to             = lambda *args, **kwargs: model if(quantized) else model.to
		model.float          = lambda *args, **kwargs: model if(quantized) else model.float
		model.half           = lambda *args, **kwargs: model if(quantized) else model.half

	@classmethod
	def _validate_params(cls, params:Dict):
		pass

	@classmethod
	def create_model(cls, *args, **kwargs):
		cls._validate_params(kwargs)
		model = timm.create_model(*args, **kwargs)
		cls._make_quantizable(model, quantized=False)
		return model

	@classmethod
	def _get_arch_key_from_save_dir(cls, save_dir:str):
		with open(BaseHQQModel.get_config_file(save_dir), "r") as file:
			config = json.load(file)
		return config['architecture']



