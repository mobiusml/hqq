#!pip install git+https://github.com/mobiusml/gemlite
#VLLM_USE_V1=0 TRITON_PRINT_AUTOTUNING=1 ipython3 ... #Make sure to disable V1!
#############################################################
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

#Set the backend
from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
#set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.MARLIN)
set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)
#set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.PYTORCH)

#Pre-quantized model
model_id = "mobiuslabsgmbh/Llama-3.1-8B-Instruct_4bitgs64_hqq_hf"

#############################################################
#On-the-fly quantization
#from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
#set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head'])
##set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic_int8', skip_modules=['lm_head'])
##set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic_fp8', skip_modules=['lm_head'])
#model_id =  "meta-llama/Llama-3.2-3B-Instruct"
#############################################################

llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.80, dtype=torch.float16) 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
outputs = llm.generate(["What is the capital of Germany?"], sampling_params)
print(outputs[0].outputs[0].text)
