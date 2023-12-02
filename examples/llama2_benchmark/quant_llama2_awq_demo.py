import torch, transformers

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = "meta-llama/Llama-2-7b-hf" 
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

#AWQ settings
######################################################################################
from awq import AutoAWQForCausalLM
import gc, time

# Load model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth, cache_dir=cache_path)
model     = AutoAWQForCausalLM.from_pretrained(model_id,         use_auth_token=hf_auth, cache_dir=cache_path, resume_download=True) 

#quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, 'version':'GEMM'}
quant_config = {"w_bit": 4, "q_group_size": 64, "zero_point": True, 'version':'GEMM'}

t1 = time.time()
model.quantize(tokenizer, quant_config=quant_config)
t2 = time.time()
print('Took ' + str(t2-t1) + ' seconds to quantize the model with AWQ')

model = model.cuda()
torch.cuda.empty_cache()
gc.collect()

#Evaluate the quantized model 
######################################################################################
from eval_model import eval_wikitext2
eval_wikitext2(model, tokenizer, verbose=True)

