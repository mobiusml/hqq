# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

import torch
from gemlite.core import GemLiteLinearTriton, DType

from ..core.quantize import HQQLinear
from ..core.peft import HQQLinearLoRA


def patch_hqq_to_gemlite(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, HQQLinear):
        hqq_layer = layer
    if isinstance(layer, HQQLinearLoRA):
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    if hqq_layer.meta["group_size"] is None:
        hqq_layer.meta["group_size"] = hqq_layer.in_features

    gemlite_linear = GemLiteLinearTriton(
        hqq_layer.meta["nbits"],
        group_size=hqq_layer.meta["group_size"],
        in_features=hqq_layer.in_features,
        out_features=hqq_layer.out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
        acc_dtype=DType.FP16,
        exhaustive=False,
    )

    orig_shape = hqq_layer.meta["shape"]
    W_q = hqq_layer.unpack(dtype=torch.uint8).view(orig_shape)
    scales = hqq_layer.meta["scale"].clone()
    zeros = hqq_layer.meta["zero"].clone()
    gemlite_linear.pack(W_q, scales, zeros, None)
    gemlite_linear.name = hqq_layer.name

    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return gemlite_linear

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = gemlite_linear

    return layer
