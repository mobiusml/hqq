# pip install git+https://github.com/mobiusml/hqq.git;
# pip install bitblas;
# num_threads=16; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3 
##########################################################################################################################################################
import torch, os

cache_path     = '.'
model_id       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
compute_dtype  = torch.float16 
device         = 'cuda:0'

##########################################################################################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
model        = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="sdpa")

#Quantize
#all 4-bit
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)

#Mixed 4-bit (bitblas) / 2-bit (ATEN)
# quant_config = {
#     "self_attn.q_proj": BaseQuantizeConfig(nbits=2, group_size=32, axis=0),
#     "self_attn.k_proj": BaseQuantizeConfig(nbits=2, group_size=32, axis=0),
#     "self_attn.v_proj": BaseQuantizeConfig(nbits=2, group_size=32, axis=0),
#     "self_attn.o_proj": BaseQuantizeConfig(nbits=2, group_size=32, axis=0),

#     "mlp.gate_proj": BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
#     "mlp.up_proj":   BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
#     "mlp.down_proj": BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
# }
# HQQLinear.set_backend(HQQBackend.ATEN)

AutoHQQHFModel.setup_model(model)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

##########################################################################################################################################################

#Replace HQQLinear layers matmuls to support int4 mm
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="bitblas", verbose=True) #It takes a while...

#Import custom HF generator
from hqq.utils.generation_hf import HFGenerator

#Generate
#gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile=None)
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() 

out = gen.generate("Write an essay about large language models.", print_tokens=True)
