# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024

import torch
import marlin
from ..core.quantize import HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA

class MarlinLinear(torch.nn.Module):
    def __init__(
        self, W: torch.Tensor, scales: torch.Tensor, u=None, bias=None, groupsize=-1
    ):
        super().__init__()

        m, n = W.shape
        device = W.device
        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        effective_groupsize = m if (groupsize == -1) else groupsize

        _layer = marlin.Layer(m, n, groupsize=groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = effective_groupsize
        _layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=device)
        _layer.s = torch.empty(
            (m // effective_groupsize, n), dtype=torch.half, device=device
        )
        _layer.pack(_linear, scales.t())

        self.bias = bias.half() if (bias is not None) else None
        self.Wq_packed = _layer.B.clone()
        self.scales = _layer.s.clone()
        self.workspace_fp = torch.zeros(n // 128 * 16, device=device)
        self.in_features = m
        self.out_features = n
        self.group_size = effective_groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        self.u = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None

        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), dtype=x.dtype, device=x.device
        )
        marlin.mul(
            x.to(self.device).view((-1, x.shape[-1])),
            self.Wq_packed,
            out.view((-1, out.shape[-1])),
            self.scales,
            self.workspace_fp,
        )
        return out

    @torch.jit.ignore
    def forward(self, x):
        out = self.matmul(x)

        if self.u is not None:
            out += torch.matmul(x.sum(axis=-1, keepdim=True), self.u)

        if self.bias is not None:
            out += self.bias

        return out

# ONLY WORKS WITH AXIS=1, group_size= - 1
def patch_hqq_to_marlin(layer, patch_params):
    hqq_layer = None
    if type(layer) is HQQLinear:
        hqq_layer = layer
    if type(layer) is HQQLinearLoRA:
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    z_shift = 8.0
    hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer

    # Check config suppport
    if (
        (hqq_layer.meta["axis"] == 0)
        or (hqq_layer.meta["group_size"] is not None)
        or (hqq_layer.meta["nbits"] != 4)
    ):
        print("Skipping marlin conversion for", getattr(hqq_layer, "name", None))
        return layer

    W_r = hqq_layer.unpack(dtype=hqq_layer.compute_dtype).t()
    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"].t()
    W_r = (W_r - z_shift) * s

    if isinstance(z, (torch.Tensor, torch.nn.Parameter)):
        z = z.t()
        u = (s * (-z + z_shift)).view([1, -1])
    else:
        u = None

    marlin_layer = MarlinLinear(W_r, s, u=u, bias=hqq_layer.bias)

    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return marlin_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = marlin_layer

    torch.cuda.empty_cache()

    return layer

