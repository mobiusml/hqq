# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

# Makes HQQ 4-bit (axis=1) compatbile with torch.ops.aten._weight_int4pack_mm

# The code is partially based on: https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py (BSD-3-Clause license)
# Only the packing logic is copied, the rest is rewritten to support HQQ's logic

# Only works with: bfloat16, axis=1.
# Only tested on Ada gpus.

import torch
import copy
from torch import bfloat16, nn, Tensor
import torch.nn.functional as F
from typing import Union

from ..core.quantize import HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA
from ..core.utils import cleanup


class HQQLinearTorchWeightOnlynt4(torch.nn.Module):
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: dict,
        del_orig: bool = True,
        compute_dtype: torch.dtype = bfloat16,
        device: str = "cuda",
        initialize: bool = True,
        inner_k_tiles=8,
        padding=True,
    ):
        super().__init__()

        self.ready = False
        self.in_gpu = False
        self.bias = None
        self.device = device
        self.compute_dtype = compute_dtype
        self.quant_config = (
            copy.deepcopy(quant_config) if (quant_config is not None) else None
        )
        self.del_orig = del_orig

        if (quant_config is None) and (linear_layer is None):
            raise Exception(
                "Invalid parameters: Both quant_config and linear_layer are None."
            )

        if (quant_config is None) and (initialize is True):
            raise Exception(
                "Invalid parameters: setting initialize to True requires a quant_config."
            )

        if self.quant_config is not None:
            weight_quant_params = self.quant_config["weight_quant_params"]
            self.groupsize = weight_quant_params["group_size"]
            self.nbits = weight_quant_params["nbits"]
            self.axis = weight_quant_params["axis"]

        if linear_layer is not None:
            self.groupsize = linear_layer.meta["group_size"]
            self.nbits = linear_layer.meta["nbits"]
            self.axis = linear_layer.meta["axis"]

        self.inner_k_tiles = inner_k_tiles
        self.padding = padding

        assert self.axis==1, "Only axis==1 is supported"
        assert self.nbits in [4], "Unsupported nbits."
        assert (
            self.compute_dtype is bfloat16
        ), "Only bfloat16 compute_dtype is supported."
        assert self.groupsize in [None, 32, 64, 128, 256], "Unsupported groupsize."
        assert self.inner_k_tiles in [2, 4, 8], "Unsupported tile."

        self.linear_layer = linear_layer

        if initialize:
            self.initialize()

    ###################### Initializers ######################
    def initialize_with_hqq_quants(self, W_q, meta, bias=None):
        self.padding = (
            False  # Force padding off, a bit tricky to post-pad with grouping
        )

        self.set_shape(meta["shape"])
        self.process_hqq_quants(W_q, meta)
        self.bias = bias
        self.ready = True
        self.in_gpu = True
        torch.cuda.empty_cache()

        return self

    def initialize(self):
        if self.linear_layer is not None:
            W = self.linear_layer.weight.data
            self.set_shape(W.shape)

            if self.in_features_diff > 0:
                W = F.pad(W, pad=(0, self.in_features_diff), value=0)

            W_q, meta = self.quantize(W, **self.quant_config)
            self.process_hqq_quants(W_q, meta)
            del W_q, meta

            self.bias = (
                None
                if (self.linear_layer.bias is None)
                else self.linear_layer.bias.to(
                    dtype=self.compute_dtype, device=self.device
                )
            )

        if self.del_orig:
            del self.linear_layer

        self.ready = True
        self.in_gpu = True
        torch.cuda.empty_cache()

        return self

    ###################### Quantize/packing ######################

    def quantize(
        self,
        W: Tensor,
        weight_quant_params: dict,
        scale_quant_params=Union[dict,None],
        zero_quant_params=Union[dict,None],
        offload_meta=False,
    ):
        W_q, meta = Quantizer.quantize(
            W,
            **weight_quant_params,
            device=self.device,
            compute_dtype=self.compute_dtype,
            bitpack=False,
        )

        # ToDO: meta quantization

        return W_q, meta

    # TODO: move these to utils
    @torch.no_grad()
    def reshape_meta_axis1(self, meta_tensor, new_group_size, shape):
        meta_tensor = meta_tensor.repeat([1, shape[1]]).reshape(shape)
        meta_tensor = torch.mean(
            meta_tensor.reshape([-1, new_group_size]), axis=1, keepdim=True
        )
        return meta_tensor

    def find_multiple(self, n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    def set_shape(self, shape):
        self.shape = shape
        self.in_features = shape[1]
        self.out_features = shape[0]

        self.origin_in_features = self.in_features
        if self.padding:
            self.in_features = self.find_multiple(self.in_features, 1024)

        self.in_features_diff = self.in_features - self.origin_in_features

    @torch.no_grad()
    def process_hqq_quants(self, W_q, meta):
        scales = meta["scale"]
        zeros = meta["zero"]
        shape = meta["shape"]

        if meta["packing"] is not None:
            W_q = Quantizer.unpack[meta["packing"]](W_q)

        if self.groupsize is None:
            self.groupsize = 128
            W_q = W_q.reshape([-1, self.groupsize])
            scales = self.reshape_meta_axis1(scales, self.groupsize, shape)
            zeros = self.reshape_meta_axis1(zeros, self.groupsize, shape)

        W_q_torch, scales_torch, zeros_torch = self.hqq_quants_to_torch_quants(
            W_q=W_q, scales=scales, zeros=zeros, shape=shape, nbits=self.nbits
        )
        self.weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
            W_q_torch, self.inner_k_tiles
        )
        self.scales_and_zeros = self.pack_scales_and_zeros(scales_torch, zeros_torch)

        del W_q_torch, scales_torch, zeros_torch
        torch.cuda.empty_cache()

    @torch.no_grad()
    def hqq_quants_to_torch_quants(
        self, W_q: Tensor, scales: Tensor, zeros: Tensor, shape, nbits=4
    ):
        W_q = W_q.to(dtype=self.compute_dtype, device=self.device)
        scales = scales.to(dtype=self.compute_dtype, device=self.device)
        zeros = zeros.to(dtype=self.compute_dtype, device=self.device)

        max_int = 2**nbits - 1
        min_int = 0
        dump = 2 ** (nbits - 1)

        # HQQ -> torch logic
        new_zeros = (scales * dump) - zeros * scales

        min_val = new_zeros - scales * dump

        # group_quantize_tensor_from_qparams
        W_r = (W_q - zeros) * scales

        W_q = (
            W_r.sub(min_val)
            .div(scales)
            .round()
            .clamp_(min_int, max_int)
            .to(torch.int32)
            .reshape(shape)
            .contiguous()
        )

        # group_dequantize_tensor_from_qparams
        # W_r = W_q*scales + min_val

        scales = scales.contiguous().reshape(shape[0], -1)
        new_zeros = new_zeros.contiguous().reshape(shape[0], -1)

        return W_q, scales, new_zeros

    def pack_scales_and_zeros(self, scales, zeros):
        assert scales.shape == zeros.shape
        assert scales.dtype == bfloat16
        assert zeros.dtype == bfloat16
        return (
            torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
            .transpose(0, 1)
            .contiguous()
        )

    ###################### Forward/matmul ######################

    # @torch.jit.ignore()
    def matmul(self, x: Tensor) -> Tensor:
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(
            x, self.weight_int4pack, self.groupsize, self.scales_and_zeros
        )
        new_shape = origin_x_size[:-1] + (self.out_features,)
        c = c.reshape(new_shape)
        return c

    # TODO without matmul
    def dequantize(self) -> Tensor:
        return self.matmul(
            torch.eye(self.in_features, dtype=self.compute_dtype, device=self.device)
        )[: self.origin_in_features].t()

    # TODO: backward
    def forward(self, x: Tensor) -> Tensor:
        if self.in_features_diff > 0:
            x = F.pad(x, pad=(0, self.in_features_diff))

        out = self.matmul(x)

        if self.bias is not None:
            out += self.bias
        return out


###################### Patching ######################
def patch_linearlayers(model, fct, patch_param=None):
    model.base_class.patch_linearlayers(
        model,
        fct,
        {lin_tag: patch_param for lin_tag in model.base_class.get_linear_tags()},
    )


def patch_hqq_to_aoint4(layer, patch_params):
    hqq_layer = None
    if type(layer) is HQQLinear:
        hqq_layer = layer
    if type(layer) is HQQLinearLoRA:
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    if hqq_layer.meta["nbits"] != 4 or hqq_layer.meta["axis"] != 1:
        print("Skipping aoint4 conversion for ", hqq_layer.name)
        return layer

    quant_config = getattr(hqq_layer, "quant_config", None)

    hqq_aoint4_layer = HQQLinearTorchWeightOnlynt4(
        None,
        quant_config=quant_config,
        compute_dtype=hqq_layer.compute_dtype,
        device=hqq_layer.device,
        del_orig=False,
        initialize=False,
        padding=False,
    )
    hqq_aoint4_layer.initialize_with_hqq_quants(
        hqq_layer.W_q, hqq_layer.meta, hqq_layer.bias
    )

    del hqq_layer
    torch.cuda.empty_cache()

    if type(layer) is HQQLinear:
        return hqq_aoint4_layer

    if type(layer) is HQQLinearLoRA:
        layer.linear_layer = hqq_aoint4_layer

    return layer


def replace_with_torchInt4(model):
    patch_linearlayers(model, patch_hqq_to_aoint4)
    cleanup()


# Force requantize, mainly to check if the padding with int4mm is faster
def patch_hqq_to_aoint4_force_requantize(layer, patch_params):
    hqq_layer = None
    if type(layer) is HQQLinear:
        hqq_layer = layer
    if type(layer) is HQQLinearLoRA:
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    if hqq_layer.meta["nbits"] != 4 or hqq_layer.meta["axis"] != 1:
        print("Skipping aoint4 conversion for ", hqq_layer.name)
        return layer

    # Create dummy linear layer to store dequantize weights
    dummy_linear = torch.nn.Linear(1, 1, bias=False)
    dummy_linear.weight.data = hqq_layer.dequantize()

    # Disable optimizer on already dequantized weights
    quant_config = hqq_layer.quant_config
    quant_config["weight_quant_params"]["optimize"] = False

    hqq_aoint4_layer = HQQLinearTorchWeightOnlynt4(
        dummy_linear,
        quant_config=quant_config,
        compute_dtype=hqq_layer.compute_dtype,
        device=hqq_layer.device,
        del_orig=True,
        initialize=True,
        padding=True,
    )

    del hqq_layer
    torch.cuda.empty_cache()

    if type(layer) is HQQLinear:
        return hqq_aoint4_layer

    if type(layer) is HQQLinearLoRA:
        layer.linear_layer = hqq_aoint4_layer

    return layer


def replace_with_torchInt4_force_requantize(model):
    patch_linearlayers(model, patch_hqq_to_aoint4_force_requantize)
    cleanup()
