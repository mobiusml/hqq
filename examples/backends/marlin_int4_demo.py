# pip install git+https://github.com/mobiusml/hqq.git;
# pip install git+https://github.com/IST-DASLab/marlin.git;
# num_threads=12; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3 
##########################################################################################################################################################
import torch, os

cache_path     = '.'
model_id       = "meta-llama/Llama-2-7b-chat-hf"
compute_dtype  = torch.float16 #int4 kernel only works with float16
device         = 'cuda:0'

##########################################################################################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import *

#Load
tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
model        = HQQModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="sdpa")

#Quantize
quant_config = BaseQuantizeConfig(nbits=4, group_size=None, quant_scale=False, quant_zero=False, axis=1)
model.quantize_model(quant_config=quant_config, compute_dtype=compute_dtype, device=device)
HQQLinear.set_backend(HQQBackend.PYTORCH)

##########################################################################################################################################################

#Replace HQQLinear layers matmuls to support int4 mm
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="marlin")

#Import custom HF generator
from hqq.utils.generation_hf import HFGenerator

#Generate
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").wamrup() 

out = gen.generate("Write an essay about large language models.", print_tokens=True)
