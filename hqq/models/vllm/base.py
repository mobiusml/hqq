# Base HQQ/VLLM model defintion
import torch
from torch import float16
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from ..base import BaseHQQModel, HQQLinear


class HQQLinearMethod(UnquantizedLinearMethod):
    def apply_weights(self, weights, x, bias=None):
        return weights["hqq_module"](x)


class BaseHQQVLLMModel(BaseHQQModel):
    @classmethod
    def quantize_model_single_worker(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device="cuda",
    ):
        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        patch_params = dict([(k, quant_config) for k in cls.get_linear_tags()])

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if quant_config is None:
                return linear_layer

            hqq_module = HQQLinear(
                linear_layer,
                quant_config,
                compute_dtype=compute_dtype,
                device=device,
                del_orig=False,
            )

            # Clear original params
            del linear_layer.weight
            linear_layer.bias = None  # bias is inside hqq_module

            # Set HQQ params
            linear_layer.linear_weights = {"hqq_module": hqq_module}
            linear_layer.linear_method = HQQLinearMethod()

            torch.cuda.empty_cache()

            return linear_layer

        cls.patch_model(
            model,
            lambda layer: layer.to(device=device, dtype=compute_dtype),
            _patch_linear,
            patch_params,
        )

    @classmethod
    def quantize_model(
        cls, model, quant_config, compute_dtype: torch.dtype = float16, device="cuda"
    ):
        workers = model.llm_engine.workers
        for i in range(len(workers)):
            cls.quantize_model_single_worker(
                workers[i].model,
                quant_config=quant_config,
                compute_dtype=compute_dtype,
                device=device,
            )

    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model_0 = model.llm_engine.workers[0].model
        model_0.config.save_pretrained(save_dir)

    @classmethod
    def cache_model_single_worker(cls, model_0, save_dir):
        model_0.config.save_pretrained(save_dir)

    @classmethod
    def serialize_weights(cls, model, verbose):
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                state_dict = module.state_dict()
                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
                else:
                    # Quantized linear layers in a VLLM model
                    if hasattr(module, "linear_weights"):
                        if "hqq_module" in module.linear_weights:
                            weights[name] = module.linear_weights[
                                "hqq_module"
                            ].state_dict()

            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights

    @classmethod
    def save_quantized(cls, model, save_dir: str, verbose: bool = False):
        model_0 = model.llm_engine.workers[0].model

        # Cache model
        cls.cache_model_single_worker(model_0, save_dir)

        # Serialization
        weights = cls.serialize_weights(model_0, verbose=verbose)

        # Save
        cls.save_weights(weights, save_dir)

    #################################################
    @classmethod
    def from_quantized_single_worker(
        cls,
        save_dir_or_hub: str,
        cache_dir: str = "",
        compute_dtype: torch.dtype = float16,
        device="cuda:0",
    ):
        # Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        # Load model from config
        model = cls.create_model(save_dir)

        # Name the layers
        cls.autoname_modules(model)

        # Load weights
        try:
            weights = cls.load_weights(save_dir)
        except Exception as error:
            print("Failed to load the weights", error)
            return

        # load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                return module.to(device=device, dtype=compute_dtype)

            state_dict = weights[module.name]
            if ("W_q" in state_dict) and ("meta" in state_dict):
                hqq_module = HQQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                hqq_module.load_state_dict(state_dict)

                # Clear original params
                del module.weight
                module.bias = None  # bias is inside hqq_module

                # Set HQQ params
                module.linear_weights = {"hqq_module": hqq_module}
                module.linear_method = HQQLinearMethod()

                torch.cuda.empty_cache()

            else:
                for key in state_dict:
                    setattr(
                        module,
                        key,
                        torch.nn.Parameter(
                            state_dict[key].to(device), requires_grad=False
                        ),
                    )

            return module

        # Load modules
        cls.patch_model(
            model,
            _load_module,
            _load_module,
            dict([(k, None) for k in cls.get_linear_tags()]),
        )
        # Load other weights that are not part of any module
        cls.post_module_load(model, weights)

        return model
