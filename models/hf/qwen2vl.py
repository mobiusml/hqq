import torch
from torch import float16, nn
from .base import AutoHQQHFModel
from os.path import join as pjoin
from ...core.peft import PeftUtils
from ...core.quantize import HQQLinear
from typing import Union
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    VisionRotaryEmbedding,
    Qwen2VLRotaryEmbedding,
)

# #Usage
# from hqq.models.hf.qwen2vl import QwenVL2HQQ
# model = QwenVL2HQQ.from_quantized('quant_model', pretrained_model_name_or_path='Qwen/Qwen2-VL-2B-Instruct', device='cuda:0', compute_dtype=torch.float16)


class QwenVL2HQQ(AutoHQQHFModel):
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation", "pretrained_model_name_or_path"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        auto_class = Qwen2VLForConditionalGeneration
        with torch.device("meta"):
            model = auto_class.from_pretrained(**model_kwargs)

        # This is broken!!!
        # config = transformers.AutoConfig.from_pretrained(
        #     cls.get_config_file(save_dir)
        # )

        # auto_class = transformers.AutoModel

        # with torch.device('meta'):
        #     model = auto_class.from_config(config, **model_kwargs)

        return model

    # Main function to load an HQQ quantized model from either HF hub or locally
    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str, None] = "",
        adapter: str = None,
        **kwargs,
    ):
        # Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        # Load model from config
        model = cls.create_model(save_dir, kwargs)

        # Track save directory
        model.save_dir = save_dir

        # Name the layers
        cls.setup_model(model)

        # Load weights
        try:
            weights = cls.load_weights(save_dir)
        except Exception:
            print("Failed to load the weights")
            raise FileNotFoundError

        # load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                if isinstance(module, VisionRotaryEmbedding):
                    module = VisionRotaryEmbedding(module.inv_freq.numel() * 2)

                if isinstance(module, Qwen2RotaryEmbedding):
                    module = Qwen2RotaryEmbedding(
                        dim=module.inv_freq.numel() * 2,
                        max_position_embeddings=model.config.max_position_embeddings,
                        base=model.config.rope_theta,
                    )

                return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            state_dict = weights[module.name]
            if "W_q" in state_dict:
                module = HQQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                module.load_state_dict(state_dict)
            else:
                for key in state_dict:
                    setattr(
                        module,
                        key,
                        nn.Parameter(
                            state_dict[key].to(
                                device=device, dtype=compute_dtype, non_blocking=True
                            ),
                            requires_grad=False,
                        ),
                    )

            return module

        # Load modules
        cls.patch_model(
            model, _load_module, _load_module, {k: None for k in model.linear_tags}
        )

        # Load other weights that are not part of any module
        cls.post_module_load(model, weights)

        model.hqq_quantized = True

        # Set base class
        model.base_class = cls

        # Add adapter
        if adapter is not None:
            try:
                PeftUtils.load_lora_weights(model, filename=pjoin(save_dir, adapter))
                PeftUtils.cast_lora_weights(model, dtype=compute_dtype)
            except Exception as e:
                print("Skipping adapter loading...", str(e))

        return model
