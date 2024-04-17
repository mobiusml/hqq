# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch
from torch import Tensor
from ..core.quantize import Quantizer

def patch_linearlayers(model, fct, patch_param=None):
    model.base_class.patch_linearlayers(model, fct, dict([(k, patch_param) for k in model.base_class.get_linear_tags()]))


#add dummy weights to a layer
def patch_add_weight_param(layer, patch_param):    
    if(hasattr(layer, 'weight') is False):
        layer.weight = torch.nn.Parameter(torch.zeros((1,), device=layer.device, dtype=layer.compute_dtype).contiguous(), requires_grad=False)
    return layer


#Makes HQQ inference compatible torch.compile fullgraph=True
def patch_hqq_simplify(layer, patch_param):
    def forward_hqq_simplified(self, x):
        out = torch.matmul(x, self.dequantize().T)
        if(self.bias is not None):
            out += self.bias 
        return out 

    if(type(layer) is HQQLinear):
        layer.forward = lambda x: forward_hqq_simplified(layer, x)
    return layer

def get_lowrank_tuple_torch_gpu(tensor, max_rank, eps=None):
    t       = tensor.t().float()
    u, s, v = torch.linalg.svd(t)
    u, s, v = u[:,:max_rank], s[:max_rank], v[:max_rank, :]
    us      = torch.matmul(u, torch.diag(s))
    A, B    = (v.t(), us.t()) #t ~ AB
    if(eps is not None):
        A  = A.clamp(min=-eps, max=eps)
        B   = B.clamp(min=-eps, max=eps)
    return A.to(tensor.dtype), B.to(tensor.dtype)


#Merges HQQ's zeros with LoRA weights. ONLY works with group_size=None and axis=1
def patch_merge_zeros_with_lora(layer, patch_params={'z_shift':8, 'keep_lora':False}):

    layer.z_shift   = patch_params['z_shift']
    layer.keep_lora = patch_params['keep_lora']
    
    shape = layer.linear_layer.meta['shape']
    z     = layer.linear_layer.meta['zero']
    s     = layer.linear_layer.meta['scale']
    u     = (s*(-z + layer.z_shift)).flatten().view([1, -1])
    
    ###########################################
    if(layer.keep_lora is False):
        A, B  = layer.lora_A.data, layer.lora_B.data
        onz   = torch.ones((shape[1], 1), device=u.device, dtype=u.dtype) 

        #Cat
        A = torch.cat([A, onz], axis=1)
        B = torch.cat([B,   u], axis=0)

        # #Re-rank
        # #A, B = get_lowrank_tuple_torch_gpu(torch.matmul(A, B) + torch.matmul(onz, u), max_rank=layer.r + 1)

        layer.lora_A.data = A.to(dtype=layer.lora_A.dtype)
        layer.lora_B.data = B.to(dtype=layer.lora_B.dtype)

    else:
        layer.u = torch.nn.Parameter(u, requires_grad=False)
    ###########################################
    layer.linear_layer.meta['zero']  = 0.

    torch.cuda.empty_cache()

    def forward_linear_updated(self, x: Tensor) -> Tensor:

        compute_dtype = self.linear_layer.compute_dtype
        meta          = self.linear_layer.meta 
        W_q           = self.linear_layer.W_q 

        W_r           = Quantizer.unpack[meta["packing"]](W_q, dtype=compute_dtype).t()
        scale         = meta['scale'].t()

        out           = torch.matmul(x, (W_r - self.z_shift))*scale #Symmetric quant
        return out

    layer.linear_layer.forward = lambda x: forward_linear_updated(layer, x)

    def forward_updated(self, x: Tensor) -> Tensor:
        out  = self.linear_layer.forward(x)

        if(layer.keep_lora):
            out += torch.matmul(x.sum(axis=-1, keepdim=True), self.u)

        out += self.forward_lora(x) + self.bias
        return out

    layer.forward = lambda x: forward_updated(layer, x)

    return layer