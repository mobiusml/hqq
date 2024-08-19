# pip install git+https://github.com/mobiusml/hqq.git;
# num_threads=12; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3 

# Tested on 4090: up to 154 tokens/sec with default compile_args
##########################################################################################################################################################
import torch, os

cache_path     = '.'
model_id       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
compute_dtype  = torch.bfloat16 #int4 kernel only works with bfloat16
device         = 'cuda:0'

##########################################################################################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
model        = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="sdpa")
AutoHQQHFModel.setup_model(model)

#Quantize
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
HQQLinear.set_backend(HQQBackend.PYTORCH)

##########################################################################################################################################################

#Replace HQQLinear layers matmuls to support int4 mm
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="torchao_int4")

#Import custom HF generator
from hqq.utils.generation_hf import HFGenerator

#Generate
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() 

out = gen.generate("Write an essay about large language models.", print_tokens=True)
