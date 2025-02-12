####################################################
from typing import Any, Dict, List, Optional
import torch
import logging

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    get_pack_factor,
)
from vllm.model_executor.parameter import BasevLLMParameter, PackedvLLMParameter

# HQQ
from vllm.model_executor.layers.quantization.hqq_marlin import (
    HQQMarlinConfig,
    HQQZeroScaleParameter,
    HQQEmptyParameter,
    error_loader,
)
from ..core.quantize import Quantizer

# Add new linear methods in order to use loader_v2
import vllm.model_executor.layers.linear as vllm_linear

HQQ_LINEAR_METHODS = ["HQQGemLiteVLLMLinear", "HQQPytorchVLLMLinear"]

for linear_method in HQQ_LINEAR_METHODS:
    if linear_method not in vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append(linear_method)

# Gemlite
try:
    from gemlite.core import DType, GemLiteLinear
    gemlite_is_available = True
except Exception:
    gemlite_is_available = False

logger = logging.getLogger(__name__)

# Hugging Face config quant name tag
QUANT_NAME = "hqq"


# Faster unpacking
@torch.compile()
def unpack_rows(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
    dtype: torch.dtype = torch.uint8,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0
    assert packed_q_w.shape == (
        size_k // pack_factor,
        size_n,
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor
    )

    packed_q_w_copy = packed_q_w.clone()
    q_res = torch.empty((size_k, size_n), dtype=dtype, device=packed_q_w.device)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_copy & mask
        packed_q_w_copy >>= num_bits
        q_res[i::pack_factor, :] = vals

    return q_res


# Override HQQweightParameter to support more nbits.
# TODO: 3-bit support not added yet.
class HQQweightParameter(PackedvLLMParameter):
    def __init__(self, packed_factor: int, packed_dim: int, weight_bits: int, **kwargs):
        super().__init__(packed_factor, packed_dim, None, **kwargs)
        self.weight_bits = weight_bits
        self.packing = Quantizer.bit_to_packing[self.weight_bits]
        self.input_shape = self.shape[self.input_dim] * self.packed_factor
        self.output_shape = self.shape[self.output_dim]

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(-1, self.input_shape).transpose(1, 0)
        loaded_weight = gptq_pack(
            loaded_weight,
            self.weight_bits,
            loaded_weight.shape[0],
            loaded_weight.shape[1],
        )
        super().load_merged_column_weight(loaded_weight, **kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(self.output_shape, -1).transpose(1, 0)
        loaded_weight = gptq_pack(
            loaded_weight,
            self.weight_bits,
            loaded_weight.shape[0],
            loaded_weight.shape[1],
        )
        super().load_row_parallel_weight(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(-1, self.input_shape).transpose(1, 0)
        loaded_weight = gptq_pack(
            loaded_weight,
            self.weight_bits,
            loaded_weight.shape[0],
            loaded_weight.shape[1],
        )
        super().load_qkv_weight(loaded_weight, **kwargs)


####################################################################################################################################
####################################################################################################################################
# Base HQQ/VLLM Linear method
class HQQBaseVLLMConfig(QuantizationConfig):
    """Config class for HQQ"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        skip_modules: Optional[List[str]] = None,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // weight_bits  # pre-packed into int32 in GPTQ format
        self.skip_modules = skip_modules
        self.packing = Quantizer.bit_to_packing[self.weight_bits]

    def __repr__(self) -> str:
        return (
            f"HQQBaseVLLMConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> str:
        return QUANT_NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HQQBaseVLLMConfig":
        wq_params = config["quant_config"]["weight_quant_params"]
        weight_bits = cls.get_from_keys(wq_params, ["nbits"])
        axis = cls.get_from_keys(wq_params, ["axis"])
        group_size = cls.get_from_keys(wq_params, ["group_size"])
        skip_modules = config["skip_modules"]

        assert axis == 1, "Only axis=1 is supported for HQQ quantized models with VLLM."
        return cls(weight_bits, group_size, skip_modules)

    def is_layer_skipped(self, prefix: str) -> bool:
        # Split the prefix into its dot-separated components
        components = prefix.split(".")

        # Check if any of the skip modules exactly matches any component
        return self.skip_modules is not None and any(
            module_name in components for module_name in self.skip_modules
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return HQQPytorchVLLMLinear(self)
        return None


class HQQBaseVLLMLinear(LinearMethodBase):
    """Base HQQ Linear VLLM method"""

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.output_size_per_partition = sum(output_partition_sizes)
        self.input_size_per_partition = input_size_per_partition

        weight_loader = extra_weight_attrs.get("weight_loader", error_loader)

        self.scales_and_zp_size = (
            input_size_per_partition // self.quant_config.group_size
        )

        #######################################################################################
        # Transposed - GPTQ/GemLited packed
        W_q = HQQweightParameter(
            data=torch.empty(
                self.input_size_per_partition // self.quant_config.pack_factor,
                self.output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_bits=self.quant_config.weight_bits,
            weight_loader=weight_loader,
        )
        layer.register_parameter("W_q", W_q)

        zero = HQQZeroScaleParameter(
            data=torch.empty(
                self.output_size_per_partition,
                self.scales_and_zp_size,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("zero", zero)

        scale = HQQZeroScaleParameter(
            data=torch.empty(
                self.output_size_per_partition,
                self.scales_and_zp_size,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("scale", scale)

        ########################################################################################
        setattr(
            layer,
            "current_shape",
            [self.output_size_per_partition, self.input_size_per_partition],
        )

        layer.register_parameter(
            "shape",
            BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int), weight_loader=weight_loader
            ),
        )
        layer.register_parameter(
            "nbits",
            BasevLLMParameter(
                data=torch.empty(1, dtype=torch.int), weight_loader=weight_loader
            ),
        )

        ignore_parameters = (
            "axis",
            "channel_wise",
            "compute_dtype",
            "encoded_state_dict",
            "group_size",
            "offload_meta",
            "optimize",
            "packing",
            "quant_scale",
            "quant_zero",
            "round_zero",
            "stores_quant_config",
            "unpack_view_dtype",
            "view_as_float",
        )

        for name in ignore_parameters:
            layer.register_parameter(
                name,
                HQQEmptyParameter(data=torch.empty(0), weight_loader=weight_loader),
            )

        #########################################################################################

    def unpack(self, layer, dtype):
        return unpack_rows(
            layer.W_q,
            num_bits=self.quant_config.weight_bits,
            size_k=self.input_size_per_partition,
            size_n=self.output_size_per_partition,
            dtype=dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Shape
        setattr(layer, "orig_shape", torch.Size(layer.shape))
        del layer.shape

        setattr(layer, "W_nbits", int(layer.nbits))
        del layer.nbits

        del layer.compute_dtype
        setattr(layer, "compute_dtype", layer.scale.dtype)


####################################################################################################################################
####################################################################################################################################
# Pytorch
class HQQPytorchConfig(HQQBaseVLLMConfig):
    """Config class for HQQ"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        skip_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__(weight_bits, group_size, skip_modules)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return HQQPytorchVLLMLinear(self)
        return None


class HQQPytorchVLLMLinear(HQQBaseVLLMLinear):
    """Linear HQQ VLLM with Pytorch backend"""

    def __init__(self, quant_config: QuantizationConfig):
        super().__init__(quant_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Repack for faster dequant
        group_size = self.quant_config.group_size
        W_q = (
            self.unpack(layer, dtype=layer.compute_dtype)
            .T.reshape(-1, group_size)
            .contiguous()
        )
        layer.W_q = torch.nn.Parameter(
            Quantizer.pack[self.quant_config.packing](W_q), requires_grad=False
        )

        torch.cuda.empty_cache()

    @torch.compile()
    def dequantize(self, layer):  # Only 8, 4, 2, 1 bit support. 3-bit NOT supported yet
        scale = layer.scale.view(-1, 1)  # non-transposed
        zero = layer.zero.view(-1, 1)  # non-transposed

        group_size = self.quant_config.group_size
        W_q = Quantizer.unpack[self.quant_config.packing](
            layer.W_q, dtype=layer.compute_dtype
        ).view(-1, group_size)
        W_r = ((W_q - zero) * scale).view(layer.current_shape)
        return W_r

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        W_r = self.dequantize(layer).T
        out = torch.matmul(x, W_r)

        if bias is not None:
            out += bias

        return out


####################################################################################################################################
####################################################################################################################################
# GemLite
class HQQGemLiteConfig(HQQBaseVLLMConfig):
    """Config class for HQQ"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        skip_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__(weight_bits, group_size, skip_modules)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return HQQGemLiteVLLMLinear(self)
        return None


class HQQGemLiteVLLMLinear(HQQBaseVLLMLinear):
    """Linear HQQ VLLM with GemLite backend"""

    gemlite_packing_bitwidth = 32

    def __init__(
        self,
        quant_config: QuantizationConfig,
    ):
        super().__init__(quant_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Gemlite
        gemlite_linear = GemLiteLinear(
            self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            in_features=self.input_size_per_partition,
            out_features=self.output_size_per_partition,
            input_dtype=DType.FP16,
            output_dtype=DType.FP16,
            scaled_activations=False,
        )

        gemlite_linear.pack(
            self.unpack(layer, dtype=torch.uint8)
            .T.contiguous()
            .view(layer.current_shape),
            layer.scale.view(-1, 1),
            layer.zero.view(-1, 1),
            bias=None,
            packing_bitwidth=HQQGemLiteVLLMLinear.gemlite_packing_bitwidth,
        )

        layer.gemlite_linear = gemlite_linear
        del layer.W_q, layer.scale, layer.zero

        torch.cuda.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = layer.gemlite_linear(x)

        if bias is not None:
            out += bias

        return out


####################################################################################################################################
####################################################################################################################################


# Allows overriding a VLLM quant method with arbitrary configs
def patch_vllm_quant_method(quant_name: str, quant_config: QuantizationConfig):
    import vllm.model_executor.layers.quantization as vllm_quantization

    get_quantization_config_orig = vllm_quantization.get_quantization_config

    def get_quantization_config_patched(quantization: str):
        if quantization == quant_name:
            return quant_config
        else:
            return get_quantization_config_orig(quantization)

    vllm_quantization.get_quantization_config = get_quantization_config_patched


class VLLM_HQQ_BACKEND:
    MARLIN = HQQMarlinConfig
    GEMLITE = HQQGemLiteConfig
    PYTORCH = HQQPytorchConfig


DEFAULT_VLLM_HQQ_BACKEND = VLLM_HQQ_BACKEND.MARLIN

def set_vllm_hqq_backend(backend: QuantizationConfig):
    global DEFAULT_VLLM_HQQ_BACKEND
    DEFAULT_VLLM_HQQ_BACKEND = backend
    if (gemlite_is_available == False and backend == VLLM_HQQ_BACKEND.GEMLITE):
        logger.error(
            "The GemLite backend is not availble. Make sure gemlite is installed: https://github.com/mobiusml/gemlite"
        )
    return patch_vllm_quant_method(QUANT_NAME, backend)


##################################################################################################################################
#Model specific patching

def patch_mixtral():
    import torch
    import torch.nn as nn
    from typing import Optional
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    import vllm.model_executor.models.mixtral_quant as mixtral_quant

    #Mixtral
    class MixtralMLPRowParallel(nn.Module):

        def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            intermediate_size: int,
            quant_config: Optional[QuantizationConfig] = None,
        ) -> None:
            super().__init__()
            self.num_experts = num_experts
            self.ffn_dim = intermediate_size
            self.hidden_dim = hidden_size

            self.w1 = RowParallelLinear(self.hidden_dim,
                                       self.ffn_dim,
                                       bias=False,
                                       quant_config=quant_config)
            self.w2 = RowParallelLinear(self.ffn_dim,
                                       self.hidden_dim,
                                       bias=False,
                                       quant_config=quant_config)
            self.w3 = RowParallelLinear(self.hidden_dim,
                                       self.ffn_dim,
                                       bias=False,
                                       quant_config=quant_config)

            # TODO: Use vllm's SiluAndMul
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            w1_out, _ = self.w1(hidden_states)
            w1_out = self.act_fn(w1_out)
            w3_out, _ = self.w3(hidden_states)
            current_hidden_states = w1_out * w3_out
            current_hidden_states, _ = self.w2(current_hidden_states)
            return current_hidden_states

    mixtral_quant.MixtralMLP = MixtralMLPRowParallel