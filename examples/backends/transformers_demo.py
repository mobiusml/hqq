#Works with multi-gpu as well, tested with BitBlas

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

device    = 'auto'
dtype     = torch.float16
model_id  = 'meta-llama/Meta-Llama-3-8B-Instruct'
cache_dir = '.' 

quant_config  = HqqConfig(nbits=4, group_size=64, axis=1)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    cache_dir=cache_dir,
    device_map=device, 
    quantization_config=quant_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

#Patching
from hqq.utils.patching import *
from hqq.core.quantize import *
HQQLinear.set_backend(HQQBackend.PYTORCH)
prepare_for_inference(model, backend='bitblas', verbose=True) #Takes a while

#Import custom HF generator
from hqq.utils.generation_hf import HFGenerator

#Generate
gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile=None) #Quick test - slower inference
#gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile="partial").warmup() #Takes a while - fastest

out = gen.generate("Write an essay about large language models.", print_tokens=True)
