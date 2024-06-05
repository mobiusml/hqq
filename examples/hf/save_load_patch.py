#nightly build
#pip uninstall torch -y; pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121;
#pip install git+https://github.com/mobiusml/hqq.git;
#pip install transformers==4.40.0 #to use the custom hugging face generator

######################################################################################################
#Quantize and save the quantized model

import torch, os, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *
from hqq.core.utils import cleanup

model_id      = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir     = ""
device        = 'cuda:0'
compute_dtype = torch.bfloat16

#####################################
# #Transformers transformers=>4.41.0
# from transformers import HqqConfig
# quant_config = HqqConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, axis=1)

# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     torch_dtype=compute_dtype, 
#     device_map=device, 
#     quantization_config=quant_config,
#     cache_dir=cache_dir
# )

#hqq lib
quant_config  = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1)
model         = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, cache_dir=cache_dir)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
###################################

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

#Save
AutoHQQHFModel.save_quantized(model, save_path)
tokenizer.save_pretrained(save_path)
del model; cleanup()

######################################################################################################
#Load a quantized model and performance optimized inference 

import torch
from transformers import AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

device        = 'cuda:0'
compute_dtype = torch.bfloat16
save_path     = 'quantized-model'

#Load
model     = AutoHQQHFModel.from_quantized(save_path, device=device, compute_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(save_path)

#Since we are working with a saved model, we need to specify the quant_config of each layer. You don't need this if you quantize on-the-fly.
from hqq.utils.patching import patch_linearlayers, patch_add_quant_config
quant_config  = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1) 
patch_linearlayers(model, patch_add_quant_config, quant_config)

# Inferece
from hqq.utils.patching import prepare_for_inference
from hqq.utils.generation_hf import HFGenerator

#Set backend: use HQQBackend.ATEN for axis=0, but doesn't work with torchao or marlin kernels
HQQLinear.set_backend(HQQBackend.PYTORCH) 

#Patch 
prepare_for_inference(model, backend="torchao_int4")

# Generate: first 5 runs are slow because of torch.compile, it's normal.
gen = HFGenerator(model, tokenizer, max_new_tokens=512, do_sample=True, compile="partial")

out = gen.generate("Write an essay about large language models.", print_tokens=False) #print tokens True sometimes produces a CUDA error.

print(out['output_text'])
