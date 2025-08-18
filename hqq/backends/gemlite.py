# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024

import torch
from ..core.quantize import HQQLinear
from ..core.peft import HQQLinearLoRA

from gemlite.helper import A16Wn

def patch_hqq_to_gemlite(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, HQQLinear):
        hqq_layer = layer
    if isinstance(layer, HQQLinearLoRA):
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    if hqq_layer.meta["group_size"] == None:
        hqq_layer.meta["group_size"] = (
            hqq_layer.in_features if (hqq_layer.meta["axis"] == 1)
            else hqq_layer.out_features
        )

    gemlite_linear = A16Wn(device=hqq_layer.device).from_hqqlinear(hqq_layer)

    if isinstance(layer, HQQLinear):
        return gemlite_linear

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = gemlite_linear

    return layer
