import torch, transformers

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = "meta-llama/Llama-2-7b-hf" 
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

#Load model on the CPU
######################################################################################
model     = transformers.AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, cache_dir=cache_path) 
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,        use_auth_token=hf_auth, cache_dir=cache_path) 

#Quantize the model
######################################################################################
from hqq.core.quantize import hqq_base_quant_config
from hqq.models.llama  import LlamaHQQ

#quant_config = hqq_base_quant_config(nbits=8, group_size=128)
quant_config = hqq_base_quant_config(nbits=4, group_size=64)
#quant_config = hqq_base_quant_config(nbits=3, group_size=64)
#quant_config = hqq_base_quant_config(nbits=2, group_size=16)
#quant_config = hqq_base_quant_config(nbits=2, group_size=16, quant_scale=True) #scale is quantized to 8-bit/g=128

LlamaHQQ.quantize_model(model, quant_config=quant_config)

# #Evaluate the quantized model 
######################################################################################
from eval_model import eval_wikitext2
eval_wikitext2(model, tokenizer, verbose=True) 

