# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch
from torch import Tensor
from ..core.quantize import Quantizer, HQQLinear
from ..core.utils import cleanup
from ..core.peft import HQQLinearLoRA
from ..models.hf.base import AutoHQQHFModel
from ..backends.torchao import patch_hqq_to_aoint4
from ..backends.marlin import patch_hqq_to_marlin


def patch_linearlayers(model, fct, patch_param=None, verbse=False):
    base_class = model.base_class if (hasattr(model, "base_class")) else AutoHQQHFModel
    base_class.setup_model(model)
    model.base_class.patch_linearlayers(
        model, fct, dict([(k, patch_param) for k in model.linear_tags]), verbose=verbse
    )


def patch_add_quant_config(layer, patch_param):
    if type(layer) is HQQLinear:
        layer.quant_config = patch_param
    if type(layer) is HQQLinearLoRA:
        layer.linear_layer.quant_config = patch_param
    return layer


# add dummy weights to a layer
def patch_add_weight_param(layer, patch_param):
    if hasattr(layer, "weight") is False:
        if hasattr(layer, "device"):
            device_ = layer.device
        else:
            param = [p for p in layer.parameters()]
            device_ = param[0].device if (len(param) > 0) else patch_param["device"]

        fp_param = [p for p in layer.parameters() if p.is_floating_point()]
        dtype_ = fp_param[0].dtype if (len(fp_param) > 0) else patch_param["dtype"]

        layer.weight = torch.nn.Parameter(
            torch.zeros((1,), device=device_, dtype=dtype_), requires_grad=False
        )
    return layer


# Optimize HQQLinear.forward for inference
def patch_hqq_inference(layer, patch_param):
    def forward_hqq_inferece(self, x):
        out = torch.matmul(x, self.dequantize().T)  # TODO GEMV use-case
        if self.bias is not None:
            out += self.bias
        return out

    if type(layer) is HQQLinear:
        layer.forward = lambda x: forward_hqq_inferece(layer, x)

    if type(layer) is HQQLinearLoRA:
        if type(layer.linear_layer) is HQQLinear:
            layer.linear_layer.forward = lambda x: forward_hqq_inferece(
                layer.linear_layer, x
            )

    return layer


# Optimize HQQLinearLoRA.forward for inference
def patch_lora_inference(layer, patch_param):
    def forward_lora_inference(self, x):
        out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        return out

    if type(layer) is HQQLinearLoRA:
        layer.forward_lora = lambda x: forward_lora_inference(layer, x)
    return layer


def prepare_for_inference(model, allow_merge=False, backend="default"):
    if backend == "torchao_int4":
        allow_merge = False

    patch_linearlayers(model, patch_hqq_inference)
    patch_linearlayers(model, patch_lora_inference)
    cleanup()

    if backend == "torchao_int4":
        patch_linearlayers(model, patch_hqq_to_aoint4)
        cleanup()
    if allow_merge:  # only compatible with symmetric quant kernels
        patch_linearlayers(
            model, patch_merge_zeros_with_lora, {"z_shift": 8, "keep_lora": False}
        )
        cleanup()
    if backend == "marlin":
        patch_linearlayers(model, patch_hqq_to_marlin, verbse=True)
        cleanup()

    patch_linearlayers(
        model, patch_add_weight_param, {"device": model.device, "dtype": model.dtype}
    )
    cleanup()


def get_lowrank_tuple_torch_gpu(tensor, max_rank, eps=None):
    t = tensor.t().float()
    u, s, v = torch.linalg.svd(t)
    u, s, v = u[:, :max_rank], s[:max_rank], v[:max_rank, :]
    us = torch.matmul(u, torch.diag(s))
    A, B = (v.t(), us.t())  # t ~ AB
    if eps is not None:
        A = A.clamp(min=-eps, max=eps)
        B = B.clamp(min=-eps, max=eps)
    return A.to(tensor.dtype), B.to(tensor.dtype)


# Merges HQQ's zeros with LoRA weights. ONLY works with group_size=None and axis=1
def patch_merge_zeros_with_lora(layer, patch_params={"z_shift": 8, "keep_lora": False}):
    if type(layer) is HQQLinearLoRA:
        # Check config suppport
        hqq_layer = layer.linear_layer
        if (hqq_layer.meta["axis"] == 0) or (hqq_layer.meta["group_size"] is not None):
            print('Skipping zeros lora merging for', layer.name)
            return layer

        layer.z_shift = patch_params["z_shift"]
        layer.keep_lora = patch_params["keep_lora"]

        shape = layer.linear_layer.meta["shape"]
        z = layer.linear_layer.meta["zero"]
        s = layer.linear_layer.meta["scale"]
        u = (s * (-z + layer.z_shift)).flatten().view([1, -1])

        ###########################################
        if layer.keep_lora is False:
            A, B = layer.lora_A.data, layer.lora_B.data
            onz = torch.ones((shape[1], 1), device=u.device, dtype=u.dtype)

            # Cat
            A = torch.cat([A, onz], axis=1)
            B = torch.cat([B, u], axis=0)

            # #Re-rank
            # #A, B = get_lowrank_tuple_torch_gpu(torch.matmul(A, B) + torch.matmul(onz, u), max_rank=layer.r + 1)

            layer.lora_A.data = A.to(dtype=layer.lora_A.dtype)
            layer.lora_B.data = B.to(dtype=layer.lora_B.dtype)

            layer.u = None

        else:
            layer.u = torch.nn.Parameter(u, requires_grad=False)
        ###########################################
        layer.linear_layer.meta["zero"] = 0.0

        torch.cuda.empty_cache()

        def forward_linear_updated(self, x: Tensor) -> Tensor:
            compute_dtype = self.linear_layer.compute_dtype
            meta = self.linear_layer.meta
            W_q = self.linear_layer.W_q

            W_r = Quantizer.unpack[meta["packing"]](W_q, dtype=compute_dtype).t()
            scale = meta["scale"].t()

            out = torch.matmul(x, (W_r - self.z_shift)) * scale  # Symmetric quant
            return out

        layer.linear_layer.forward = lambda x: forward_linear_updated(layer, x)

        def forward_updated(self, x: Tensor) -> Tensor:
            out = self.linear_layer.forward(x)

            if self.u is not None:
                out += torch.matmul(x.sum(axis=-1, keepdim=True), self.u)

            out += self.forward_lora(x) + self.bias
            return out

        layer.forward = lambda x: forward_updated(layer, x)

    return layer
