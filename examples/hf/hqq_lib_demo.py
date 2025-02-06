# pip install git+https://github.com/mobiusml/hqq.git;
# pip install git+https://github.com/mobiusml/gemlite.git; #to use the gemlite backend
# pip install bitblas; #to use the bitblas backend
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 ipython3;
########################################################################
import torch
device        = 'cuda:0'
backend       = "torchao_int4" #'torchao_int4' #"torchao_int4" (4-bit only) or "bitblas" (4-bit + 2-bit) or "gemlite" (8-bit, 4-bit, 2-bit, 1-bit)
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = None
model_id      = 'meta-llama/Meta-Llama-3-8B-Instruct'
########################################################################
#Load model
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", cache_dir=cache_dir)

#Quantize
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Optimize
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend=backend, verbose=True)

#Load GemLite cache
if(backend == 'gemlite'):
	import gemlite
	gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
	gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

########################################################################
##Inference Using a custom hqq generator - fastest solution + supports real-time token printing
from hqq.utils.generation_hf import HFGenerator
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() 

# gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial", 
# 									compile_options={"mode": "max-autotune-no-cudagraphs", "fullgraph": True}
# 									).enable_cuda_graph()

out = gen.generate("Write an essay about large language models.", print_tokens=True)

########################################################################
# #Inference with model,generate()
# from hqq.utils.generation_hf import patch_model_for_compiled_runtime

# patch_model_for_compiled_runtime(model, tokenizer) 

# prompt  = "Write an essay about large language models."
# inputs  = tokenizer.apply_chat_template([{"role":"user", "content":prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
# outputs = model.generate(**inputs.to(model.device), max_new_tokens=1000, cache_implementation="static", pad_token_id=tokenizer.pad_token_id) 
# #print(tokenizer.decode(outputs[0])

########################################################################
#Save gemlite cache
if(backend == 'gemlite'):
	gemlite.core.GemLiteLinear.cache_config('/tmp/gemlite_config.json') 