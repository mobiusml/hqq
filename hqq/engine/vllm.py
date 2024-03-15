# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################

# Import VLLM architectures here with the dummy_load trick
import torch
from torch import float16
import transformers
from ..models.vllm.llama import LlamaForCausalLM, LlamaHQQ
from ..models.base import BaseHQQModel
from .base import HQQWrapper
from termcolor import colored

# Set them in the model registry
import vllm

vllm.model_executor.model_loader._MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM

_HQQ_REGISTRY = {}
_HQQ_REGISTRY["LlamaForCausalLM"] = LlamaHQQ

# VLLM requires model input as string, so this is the fallback for a dummy init used to load the model from quantized weights + load the tokenizer
_ARCH_TO_DEFAULT = {}
_ARCH_TO_DEFAULT["LlamaForCausalLM"] = "meta-llama/Llama-2-7b-chat-hf"

_Parent = vllm.entrypoints.llm.LLM


# Similar to the from vllm import LLM class, with some extra parameters and only loads a dummy model
class HQQLLM(_Parent, HQQWrapper):
    _HQQ_REGISTRY = _HQQ_REGISTRY
    INIT_CACHE_MEM = 1.0  # Default cache mem in GB. This is extra-memory, but you need it otherwise runtime is slower

    def __init__(self, *args, **kwargs):
        self._validate_params(kwargs)

        if "force_skip" in kwargs:
            force_skip = kwargs["force_skip"]
            del kwargs["force_skip"]
        else:
            force_skip = False

        # This will init a dummy model
        super().__init__(*args, **kwargs)

        cache_dir = kwargs["cache_dir"] if ("cache_dir" in kwargs) else ""
        model_id = kwargs["model"]

        # Here we load the model on CPU so that we can quantize it while avoiding extra GPU cost
        self.arch_key = self.llm_engine.model_config.hf_config.architectures[0]
        model_config = self.llm_engine.model_config.hf_config
        workers = self.llm_engine.workers

        if force_skip is False:
            for i in range(len(workers)):
                workers[i].model.__init__(model_config, dummy_load=False)
                workers[i].model.load_weights(
                    model_name_or_path=model_id, cache_dir=cache_dir
                )

        self.hqq_quantized = False

    def _validate_params(self, kwargs: dict):
        if "gpu_memory_utilization" not in kwargs:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            kwargs["gpu_memory_utilization"] = HQQLLM.INIT_CACHE_MEM / total_gpu_mem

    # In case the user wants to use the fp16 model and skip quantization
    def cuda(self):
        self._check_if_already_quantized(self)
        workers = self.llm_engine.workers
        for i in range(len(workers)):
            workers[i].model = workers[i].model.half().cuda(i)
        return self

    def quantize_model(
        self, quant_config: dict, compute_dtype: torch.dtype = float16, device="cuda"
    ):
        return self.quantize_model_(
            model=self,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
        )

    def save_quantized(self, save_dir: str):
        return self.save_quantized_(model=self, save_dir=save_dir)

    @classmethod
    def _get_arch_key_from_save_dir(cls, save_dir: str):
        config = transformers.AutoConfig.from_pretrained(save_dir)
        return config.architectures[0]

    # This requires custom loading because VLLM requires a str input model to initalize the workers
    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub: str,
        compute_dtype: torch.dtype = float16,
        cache_dir: str = "",
        tensor_parallel_size: int = 1,
    ):
        assert tensor_parallel_size == 1, "Only single GPU is supported."
        # Both local and hub-support
        save_dir = BaseHQQModel.try_snapshot_download(save_dir_or_hub)
        arch_key = cls._get_arch_key_from_save_dir(save_dir)
        cls._check_arch_support(arch_key)

        # Trick to initialize the tokenizer and a dummy model inside a VLLM instance
        instance = cls(
            model=_ARCH_TO_DEFAULT[arch_key], force_skip=True, tensor_parallel_size=1
        )
        workers = instance.llm_engine.workers
        for i in range(len(workers)):
            workers[i].model = cls._get_hqq_class(
                arch_key
            ).from_quantized_single_worker(
                save_dir_or_hub=save_dir_or_hub,
                compute_dtype=compute_dtype,
                cache_dir=cache_dir,
                device="cuda:" + str(i),
            )

        cls._make_quantizable(instance, quantized=True)
        return instance


# From https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/llms/vllm.py


try:
    from langchain.llms import VLLM as LangchainVLLMBase
    from langchain_core.pydantic_v1 import root_validator

    class LangchainVLLM(LangchainVLLMBase):
        def set(self, model):
            self.client = model
            return self

        @root_validator()
        def validate_environment(cls, values: dict) -> dict:
            return values
except Exception:
    LangchainVLLM = None
    print(
        colored(
            'Langchain not installed. You can install it via "pip install langchain"',
            "cyan",
        )
    )
