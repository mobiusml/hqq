# pip install git+https://github.com/mobiusml/hqq.git;
# pip install bitblas #to use the bitblas backend
# OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 
########################################################################
import torch
device        = 'cuda:0'
backend       = "torchao_int4" #'torchao_int4' #"torchao_int4" (4-bit only) or "bitblas" (4-bit + 2-bit) or "gemlite" (8-bit, 4-bit, 2-bit, 1-bit)
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = '.' 
model_id      = 'meta-llama/Meta-Llama-3-8B-Instruct'

########################################################################
#Load model
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model     = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="sdpa")

#Quantize
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
HQQLinear.set_backend(HQQBackend.PYTORCH)

#Optimize
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend=backend, verbose=True)

#Load GemLite cache
if(backend == 'gemlite'):
	import gemlite
	gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
	gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

#Inference
########################################################################
# #Using a custom generator
# from hqq.utils.generation_hf import HFGenerator
# gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() 

# out = gen.generate("Write an essay about large language models.", print_tokens=True)
########################################################################
#Using HF model.generate()
from hqq.utils.generation_hf import patch_model_for_compiled_runtime

patch_model_for_compiled_runtime(model, tokenizer, warmup=True)

# Prompt
system_prompt = None #"You are a helpful assistant."
prompt        = "Write an essay about large language models."

messages  = [] if(system_prompt is None) else [{"role": "system", "content": system_prompt}]
messages += [{"role": "user", "content": prompt},]

inputs = tokenizer([tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=1000, cache_implementation="static", pad_token_id=tokenizer.pad_token_id) 
#print(tokenizer.decode(outputs[0]))

########################################################################
#Save gemlite cache
if(backend == 'gemlite'):
	gemlite.core.GemLiteLinear.cache_config('/tmp/gemlite_config.json') 