# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import os
import gc
import torch
from torch import nn
from torch import float16
from os.path import join as pjoin
from typing import Callable
from tqdm import tqdm
from abc import abstractmethod
from functools import partial
from typing import Union

from huggingface_hub import snapshot_download
from ..core.quantize import HQQLinear
from ..core.peft import PeftUtils

# Defined what is qualified as "linear layer"
_QUANT_LAYERS = [nn.Linear]
_IGNORE_LINEAR = ["lm_head"]


# Cleanup GPU vram
def cleanup() -> None:
    torch.cuda.empty_cache()
    gc.collect()


# Finds the parent of a node module named "name"
def find_parent(model, name: str) -> nn.Module:
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


# checks if a module is a leaf: doesn't have another module inside
def is_leaf_module(module) -> bool:
    return len(module._modules) == 0


# Get the linear_tag from a modul name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name: str) -> str:
    return ".".join(
        [
            n
            for n in name.split(".")
            if ((n not in ["model", "layers"]) and (not n.isnumeric()))
        ]
    )


# returns all children nodes from model
def get_all_children_from_model(model, ignore: list = []) -> list:
    tags = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and (name.split(".")[-1] not in ignore):
            tags.append(name)
    return tags


# Get all linear tags available
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)


def forward_device_hooked(self, *args, **kwargs):
    args = list(args)

    #eddit this to make torch.compile compatible
    for i in range(len(args)):
        if(isinstance(args[i], (torch.Tensor, torch.nn.Parameter))):#if hasattr(args[i], "to"):
            args[i] = args[i].to(self.device)

    for i in kwargs:
        if(isinstance(kwargs[i], (torch.Tensor, torch.nn.Parameter))):#if hasattr(kwargs[i], "to"):
            kwargs[i] = kwargs[i].to(self.device)

    # return self.__class__.forward(self, *args, **kwargs)
    return self.forward_orig(*args, **kwargs)


# Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function.
class BasePatch:
    # Override these OR override the main patch_model() function
    ############################################
    # This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_nonlinearlayers(
        cls, model, patch_fct: Callable, verbose: bool = True
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) not in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name]),
            )

    # This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(
        cls, model, patch_fct: Callable, patch_params: Union[dict, None], verbose: bool = True
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name], patch_param),
            )

    ############################################
    # These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod
    def set_auto_linear_tags(cls, model, ignore: list = _IGNORE_LINEAR) -> None:
        #if len(cls.get_linear_tags()) == 0:
        cls.linear_tags = get_linear_tags_from_model(model, ignore=ignore)
        cls.get_linear_tags = lambda: cls.linear_tags

    # Returns the current linear tags
    @classmethod
    def get_linear_tags(cls) -> list:
        return []

    @classmethod
    def get_ignore_layers(cls, model) -> list:
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)

    # Autmatically name modules. This is very important to save/load the weights
    @classmethod
    def autoname_modules(cls, model) -> None:
        for name, module in model.named_modules():
            module.name = name

    # Freeze all layers
    @classmethod
    def freeze_model(cls, model) -> None:
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

    # Main patching function
    @classmethod
    def patch_model(
        cls,
        model,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        cleanup()


class BaseHQQModel:
    # Override these
    ############################################
    # This method creates and empty model based on the specfied architecture
    @abstractmethod
    def create_model(self):
        pass

    # This method saves the model architecture only without inculding the weights (for example to a config.json)
    @abstractmethod
    def cache_model(cls, model, save_dir: str):
        pass

    ############################################

    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    # Save weights to disk
    @classmethod
    def save_weights(cls, weights: dict, save_dir: str) -> None:
        torch.save(weights, cls.get_weight_file(save_dir))

    # Load weights from disk
    @classmethod
    def load_weights(cls, save_dir: str, map_location = None):
        return torch.load(cls.get_weight_file(save_dir), map_location=map_location)

    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda", 
    ):
        # Set linear tags automatically
        cls.autoname_modules(model)
        cls.set_auto_linear_tags(model)

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in cls.get_linear_tags()) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in cls.get_linear_tags()}
            patch_params.update(quant_config)
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in cls.get_linear_tags()}

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.layers)
                if hasattr(model, "model")
                else len(model.layers)
            )
            all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            current_device = device_map[linear_layer.name]

            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        # Sync
        torch.cuda.synchronize()

        return model

    # Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                state_dict = module.state_dict()
                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights

    # Main function to save a quantized model
    @classmethod
    def save_quantized(cls, model, save_dir: str, verbose: bool = False):
        # Save config
        cls.cache_model(model, save_dir)

        # Serialization
        weights = cls.serialize_weights(model, verbose=verbose)

        # Save
        cls.save_weights(weights, save_dir)

    @classmethod
    def try_snapshot_download(cls, save_dir_or_hub: str, cache_dir: Union[str, None]= ""):
        if cache_dir is None:
            save_dir = pjoin("", save_dir_or_hub)
        else:
            save_dir = pjoin(cache_dir, save_dir_or_hub)

        if not os.path.exists(save_dir):
            save_dir = snapshot_download(repo_id=save_dir_or_hub, cache_dir=cache_dir)
            save_dir = pjoin(save_dir)

        # Check
        if not os.path.exists(cls.get_weight_file(save_dir)):
            raise Exception("Weight file missing. Check your cache directory.")
        if not os.path.exists(cls.get_config_file(save_dir)):
            raise Exception("Config file missing. Check your cache directory.")

        return save_dir

    # This method is specfically designed in case we need to load some weights that are not part of any module
    @classmethod
    def post_module_load(cls, model, weights: dict):
        pass

    # Main function to load an HQQ quantized model from either HF hub or locally
    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str,None] = "",
        adapter: str = None,
    ):
        # Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        # Load model from config
        model = cls.create_model(save_dir)

        # Track save directory
        model.save_dir = save_dir

        # Name the layers
        cls.autoname_modules(model)

        # Set linear tags automatically
        cls.set_auto_linear_tags(model)

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
                return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            state_dict = weights[module.name]
            if ("W_q" in state_dict) and ("meta" in state_dict):
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
            model, _load_module, _load_module, {k: None for k in cls.get_linear_tags()}
        )

        # Load other weights that are not part of any module
        cls.post_module_load(model, weights)

        # Set base class
        model.base_class = cls

        # Add adapter
        if adapter is not None:
            try:
                PeftUtils.load_lora_weights(model, filename=pjoin(save_dir, adapter))
                PeftUtils.cast_lora_weights(model, dtype=compute_dtype)
            except Exception as e:
                print("Skipping adapter loading...", str(e))

        # Sync
        torch.cuda.synchronize()

        return model
