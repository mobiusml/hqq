####################################################
from typing import Any, Dict, List, Optional
import torch, numpy, logging

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack, get_pack_factor
from vllm.model_executor.parameter import BasevLLMParameter,PackedvLLMParameter
from vllm.scalar_type import scalar_types
from vllm.model_executor.utils import set_weight_attrs

#HQQ
from vllm.model_executor.layers.quantization.hqq_marlin import HQQMarlinConfig, HQQZeroScaleParameter, HQQEmptyParameter, error_loader
from ..core.quantize import Quantizer

#Gemlite
try:
    from gemlite.core import DType, GemLiteLinear
except:
    GemLiteLinear = None

logger = logging.getLogger(__name__)

#Hugging Face config quant name tag
QUANT_NAME = 'hqq'

#Faster unpacking 
def unpack_rows(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
    dtype: torch.dtype = torch.uint8,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0
    assert packed_q_w.shape == (size_k // pack_factor, size_n), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor)


    packed_q_w_copy = packed_q_w.clone()
    q_res = torch.empty((size_k, size_n), dtype=dtype, device=packed_q_w.device)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_copy & mask
        packed_q_w_copy >>= num_bits
        q_res[i::pack_factor, :] = vals

    q_res = q_res.contiguous()

    return q_res

#Override HQQweightParameter to support more nbits.
#TODO: 3-bit support not added yet. 
class HQQweightParameter(PackedvLLMParameter):

    def __init__(self, packed_factor: int, packed_dim: int, weight_bits: int, **kwargs):
        super().__init__(packed_factor, packed_dim, None, **kwargs)
        self.weight_bits  = weight_bits
        self.packing      = Quantizer.bit_to_packing[self.weight_bits]
        self.input_shape  = self.shape[self.input_dim] * self.packed_factor
        self.output_shape = self.shape[self.output_dim]

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(-1, self.input_shape).transpose(1, 0)
        loaded_weight = gptq_pack(loaded_weight, self.weight_bits, loaded_weight.shape[0], loaded_weight.shape[1])
        super().load_merged_column_weight(loaded_weight, **kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(self.output_shape, -1).transpose(1, 0)
        loaded_weight = gptq_pack(loaded_weight, self.weight_bits, loaded_weight.shape[0], loaded_weight.shape[1])
        super().load_row_parallel_weight(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        loaded_weight = Quantizer.unpack[self.packing](loaded_weight, dtype=torch.uint8)
        loaded_weight = loaded_weight.reshape(-1, self.input_shape).transpose(1, 0)
        loaded_weight = gptq_pack(loaded_weight, self.weight_bits, loaded_weight.shape[0], loaded_weight.shape[1])
        super().load_qkv_weight(loaded_weight, **kwargs)


class HQQGemLiteConfig(QuantizationConfig):
    """Config class for HQQ"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        skip_modules: Optional[List[str]] = None,
    ) -> None:

        self.weight_bits  = weight_bits
        self.group_size   = group_size
        self.pack_factor  = 32 // weight_bits  # pre-packed into int32 in GPTQ format
        self.skip_modules = skip_modules

    def __repr__(self) -> str:
        return (f"HQQGemLiteConfig(weight_bits={self.weight_bits}, "f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> str:
        return QUANT_NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16] #torch.bfloat16

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HQQGemLiteConfig":
        wq_params = (config["quant_config"]["weight_quant_params"])
        weight_bits = cls.get_from_keys(wq_params, ["nbits"])
        group_size = cls.get_from_keys(wq_params, ["group_size"])
        skip_modules = config["skip_modules"]
        return cls(weight_bits, group_size, skip_modules)

    def is_layer_skipped(self, prefix: str) -> bool:
        # Split the prefix into its dot-separated components
        components = prefix.split('.')

        # Check if any of the skip modules exactly matches any component
        return self.skip_modules is not None and any(
            module_name in components for module_name in self.skip_modules)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return HQQMarlinMethod(self)
        return None


#GemLite
class HQQMarlinMethod(LinearMethodBase):
    """Linear HQQ
    """

    def __init__(
        self,
        quant_config: QuantizationConfig
    ):
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
        self.input_size_per_partition  = input_size_per_partition

        weight_loader = extra_weight_attrs.get("weight_loader", error_loader)

        self.scales_and_zp_size = (input_size_per_partition // self.quant_config.group_size)

        #######################################################################################
        in_features  = self.input_size_per_partition
        out_features = self.output_size_per_partition
        group_size   = self.quant_config.group_size

        #Transposed - GPTQ/GemLited packed
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
            weight_loader=weight_loader)
        layer.register_parameter("W_q", W_q)

        zero  = HQQZeroScaleParameter(data=torch.empty(self.output_size_per_partition, self.scales_and_zp_size, dtype=params_dtype), 
                                      input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("zero", zero)

        scale = HQQZeroScaleParameter(data=torch.empty(self.output_size_per_partition, self.scales_and_zp_size, dtype=params_dtype), 
                                      input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("scale", scale)

        ########################################################################################
        setattr(layer, 'current_shape', [self.output_size_per_partition, self.input_size_per_partition])

        layer.register_parameter('shape', BasevLLMParameter(data=torch.empty(2, dtype=torch.int), weight_loader=weight_loader))
        layer.register_parameter('nbits', BasevLLMParameter(data=torch.empty(1, dtype=torch.int), weight_loader=weight_loader))

        ignore_parameters = ("axis", "channel_wise", "compute_dtype", 
                             "encoded_state_dict", "group_size",
                             "offload_meta", "optimize", "packing",
                             "quant_scale", "quant_zero", "round_zero",
                             "stores_quant_config", "unpack_view_dtype", 
                             "view_as_float")

        for name in ignore_parameters:
            layer.register_parameter(name, HQQEmptyParameter(data=torch.empty(0), weight_loader=weight_loader))

        #########################################################################################


    def unpack(self, layer, dtype):
        return unpack_rows(layer.W_q, num_bits=self.quant_config.weight_bits, size_k=self.input_size_per_partition, size_n=self.output_size_per_partition, dtype=dtype)

    def dequantize(self, layer):

        scale = layer.scale.view(-1, 1) #non-transposed
        zero  = layer.zero.view(-1, 1)  #non-transposed

        group_size = self.quant_config.group_size
        W_q        = self.unpack(layer, dtype=zero.dtype).T.reshape(-1, group_size)
        W_r        = ((W_q - zero) * scale).reshape(layer.current_shape).contiguous()
        return W_r

    def apply_dequantize_on_the_fly(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        W_r = self.dequantize(layer).T
        out = torch.matmul(x, W_r)

        if(bias is not None):
            out += bias

        return out

    def apply_gemlite(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        out = layer.gemlite_linear(x)

        if(bias is not None):
            out += bias

        return out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        #Shape
        setattr(layer, 'orig_shape', torch.Size(layer.shape))
        del layer.shape
        
        setattr(layer, 'W_nbits', int(layer.nbits))
        del layer.nbits

        ###########################################################################
        # #Dequantize
        # layer.W_q.data = self.dequantize(layer)
        # del layer.scale, layer.zero

        ###########################################################################
        #Gemlite
        gemlite_linear = GemLiteLinear(
            self.quant_config.weight_bits,
            group_size=self.quant_config.group_size, 
            in_features=self.input_size_per_partition, 
            out_features=self.output_size_per_partition, 
            input_dtype=DType.FP16, 
            output_dtype=DType.FP16, 
            scaled_activations=False, 
        )

        gemlite_linear.pack(self.unpack(layer, dtype=torch.uint8).T.contiguous().view(layer.current_shape), 
                            layer.scale.view(-1, 1), 
                            layer.zero.view(-1, 1), 
                            bias=None, 
                            packing_bitwidth=32,
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

        #return self.apply_dequantize_on_the_fly(layer, x, bias)
        return self.apply_gemlite(layer, x, bias)

#Allows overriding a VLLM quant method with arbitrary configs
def patch_vllm_quant_method(quant_name: str, quant_config: QuantizationConfig):
    import vllm.model_executor.layers.quantization as vllm_quantization
    get_quantization_config_orig = vllm_quantization.get_quantization_config

    def get_quantization_config_patched(quantization: str):
        if(quantization == quant_name):
            return quant_config
        else:
            return get_quantization_config_orig(quantization)

    vllm_quantization.get_quantization_config = get_quantization_config_patched

class VLLM_HQQ_BACKEND:
    MARLIN  = HQQMarlinConfig
    GEMLITE = HQQGemLiteConfig

def set_vllm_hqq_backend(backend: QuantizationConfig = VLLM_HQQ_BACKEND.MARLIN):
    if(GemLiteLinear is None):
        logger.error('The GemLite backend is not availble. Make sure gemlite is installed: https://github.com/mobiusml/gemlite')
    return patch_vllm_quant_method(QUANT_NAME, backend)
