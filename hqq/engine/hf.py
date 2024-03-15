# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################

import transformers
import torch
from .base import HQQWrapper

from ..models.hf.llama import LlamaHQQ
from hqq.models.hf.mixtral import MixtralHQQ
from hqq.models.hf.phi import PhiHQQ
from hqq.models.hf.mistral import MistralHQQ

_HQQ_REGISTRY = {}
_HQQ_REGISTRY["LlamaForCausalLM"] = LlamaHQQ
_HQQ_REGISTRY["MixtralForCausalLM"] = MixtralHQQ
_HQQ_REGISTRY["PhiForCausalLM"] = PhiHQQ
_HQQ_REGISTRY["MistralForCausalLM"] = MistralHQQ

# Alias
AutoTokenizer = transformers.AutoTokenizer

# Used to call super() on classmethods
_Parent = transformers.AutoModelForCausalLM


class HQQModelForCausalLM(_Parent, HQQWrapper):
    _HQQ_REGISTRY = _HQQ_REGISTRY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _make_quantizable(cls, model, quantized: bool) -> None:
        model.hqq_quantized = quantized
        model.arch_key = model.config.architectures[0]
        model.quantize_model = (
            lambda quant_config,
            compute_dtype=torch.float16,
            device="cuda": cls.quantize_model_(
                model=model,
                quant_config=quant_config,
                compute_dtype=compute_dtype,
                device=device,
            )
        )
        model.save_quantized = lambda save_dir: cls.save_quantized_(
            model=model, save_dir=save_dir
        )
        model.cuda = lambda *args, **kwargs: model if (quantized) else model.cuda
        model.to = lambda *args, **kwargs: model if (quantized) else model.to
        model.float = lambda *args, **kwargs: model if (quantized) else model.float
        model.half = lambda *args, **kwargs: model if (quantized) else model.half
        model.base_class = cls._get_hqq_class(model)

    # Force loading the model on CPU and unquantized
    @classmethod
    def _validate_params(cls, params: dict) -> None:
        for p in ["load_in_4bit", "load_in_8bit"]:  # ignore these
            if p in params:
                params[p] = False
        params["device_map"] = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls._validate_params(kwargs)
        model = super(_Parent, cls).from_pretrained(*args, **kwargs)
        cls._make_quantizable(model, quantized=False)
        return model

    @classmethod
    def _get_arch_key_from_save_dir(cls, save_dir: str):
        config = transformers.AutoConfig.from_pretrained(save_dir)
        return config.architectures[0]
