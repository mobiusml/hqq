#pip install git+https://github.com/mobiusml/hqq.git #master branch latest version
#pip install bitblas #for using the bitblas backend 
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3  #change this based on your machine
###################################################
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import *
from hqq.core.quantize import *
from hqq.utils.generation_hf import HFGenerator

#Load the model
###################################################
model_id      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
compute_dtype = torch.bfloat16 #bfloat16 for torchao, float16 for bitblas
device        = 'cuda:0'
cache_dir     = '.'
model         = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="sdpa")
tokenizer     = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, device=device, compute_dtype=compute_dtype)

#Use optimized inference kernels
###################################################
HQQLinear.set_backend(HQQBackend.PYTORCH)
#prepare_for_inference(model) #default backend
prepare_for_inference(model, backend="torchao_int4") 
#prepare_for_inference(model, backend="bitblas") #takes a while to init...

# #Generate via custom HFGenerator
# ###################################################
#For longer context, make sure to allocate enough cache via the cache_size= parameter 
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() #Warm-up takes a while

gen.generate("Write an essay about large language models", print_tokens=True)
gen.generate("Tell me a funny joke!", print_tokens=True)
gen.generate("How to make a yummy chocolate cake?", print_tokens=True)

# #Generate via HF model.generate() - doesn't support print_tokens=True
# ####################################################
# model.generation_config.cache_implementation = "static"
# gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="full").warmup() #compiles the whole model

# out = gen.generate_("Write an essay about large language models") #this is calling model.generate()
