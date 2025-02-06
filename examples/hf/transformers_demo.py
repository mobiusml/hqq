# pip install git+https://github.com/mobiusml/hqq.git;
# pip install git+https://github.com/mobiusml/gemlite.git; #to use the gemlite backend
# pip install bitblas #to use the bitblas backend
# OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 
########################################################################
import torch
device        = 'cuda:0'
backend       = 'torchao_int4' #'torchao_int4' #"torchao_int4" (4-bit only) or "bitblas" (4-bit + 2-bit) or "gemlite" (8-bit, 4-bit, 2-bit, 1-bit)
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = None
model_id      = 'meta-llama/Meta-Llama-3-8B-Instruct' #Raw
#model_id      = 'mobiuslabsgmbh/Meta-Llama-3-8B-Instruct_4bitgs64_hqq_hf' #Pre-quantized

is_prequantized = 'hqq_hf' in model_id
########################################################################
#Load model
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig 

quant_config = None if(is_prequantized) else HqqConfig(nbits=4, group_size=64, axis=1)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=compute_dtype, 
    cache_dir=cache_dir,
    device_map=device, 
     attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    quantization_config=quant_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

#Save model before patching
# model.save_pretrained(saved_quant_model)
# tokenizer.save_pretrained(saved_quant_model)

#Patching
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend=backend, verbose=True) 

#Load GemLite cache
if(backend == 'gemlite'):
    import gemlite
    gemlite.core.GEMLITE_TRITON_RESTRICT_M = True
    gemlite.core.GemLiteLinear.load_config('/tmp/gemlite_config.json')

########################################################################
# ##Inference Using a custom hqq generator - currently manual compile breaks with pre-quantized llama models :(
# from hqq.utils.generation_hf import HFGenerator
# gen = HFGenerator(model, tokenizer, max_new_tokens=1000, do_sample=True, compile=False).enable_cuda_graph() 

# out = gen.generate("Write an essay about large language models.", print_tokens=True)

########################################################################
#Inference with model,generate()
from hqq.utils.generation_hf import patch_model_for_compiled_runtime

patch_model_for_compiled_runtime(model, tokenizer) 

prompt  = "Write an essay about large language models."
inputs  = tokenizer.apply_chat_template([{"role":"user", "content":prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1000, cache_implementation="static", pad_token_id=tokenizer.pad_token_id) 
#print(tokenizer.decode(outputs[0])

########################################################################
#Save gemlite cache
if(backend == 'gemlite'):
    gemlite.core.GemLiteLinear.cache_config('/tmp/gemlite_config.json') 