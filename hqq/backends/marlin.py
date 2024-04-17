# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch 
import marlin
from ..core.quantize import Quantizer

class MarlinLinear(torch.nn.Module):
    def __init__(self, W: torch.Tensor, scales: torch.Tensor, bias=None, groupsize=-1):
        super().__init__()

        m, n   = W.shape
        device = W.device
        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        effective_groupsize = m if (groupsize==-1) else groupsize

        _layer   = marlin.Layer(m, n, groupsize=groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = effective_groupsize
        _layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=device)
        _layer.s = torch.empty((m // effective_groupsize, n), dtype=torch.half, device=device)
        _layer.pack(_linear, scales.t())


        self.bias          = bias.half() if (bias is not None) else None
        self.Wq_packed     = _layer.B.clone()
        self.scales        = _layer.s.clone()
        self.workspace_fp  = torch.zeros(n // 128 * 16, device=device)
        self.in_features   = m
        self.out_features  = n
        self.group_size    = effective_groupsize
        self.axis          = 1
        self.device        = device 
        self.compute_dtype = torch.float16

        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.jit.ignore
    def matmul(self, x):
        out = torch.empty(x.shape[:-1] + (self.scales.shape[1],), dtype=x.dtype, device=x.device)
        marlin.mul(x.view((-1, x.shape[-1])), self.Wq_packed, out.view((-1, out.shape[-1])), self.scales, self.workspace_fp)
        return out

    def forward(self, x):
        out = self.matmul(x)
        if(self.bias is not None):
            out += self.bias
        return out

#ONLY WORKS WITH AXIS=1, group_size= - 1
def patch_hqq_to_marlin(layer, patch_params=None):
    z_shift   = 8.
    hqq_layer = layer.linear_layer if hasattr(layer, 'linear_layer') else layer

    W_r       = Quantizer.unpack[hqq_layer.meta['packing']](hqq_layer.W_q, dtype=hqq_layer.compute_dtype).t()
    scales    = hqq_layer.meta['scale'].t()
    W_r       = (W_r - z_shift) * scales

    #forward equivalent to: torch.matmul(x, (W_r - z_shift) * scales)

    #TODO: ADD rank-1 matmul for -zeros*scales

    marlin_layer = MarlinLinear(W_r, scales, bias=hqq_layer.bias)

    del W_r, scales

    if hasattr(layer, 'linear_layer'):
        del layer.linear_layer
        layer.linear_layer = marlin_layer
    else:
        del hqq_layer
        layer = marlin_layer

    torch.cuda.empty_cache()

    return layer
