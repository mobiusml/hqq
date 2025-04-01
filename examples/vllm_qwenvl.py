# git clone https://github.com/vllm-project/vllm.git; cd vllm; 
# VLLM_USE_PRECOMPILED=1 pip install --editable .; cd ..;
# VLLM_USE_V1=0 TRITON_PRINT_AUTOTUNING=1 ipython3 ...
##########################################################################
import torch
from PIL import Image
from urllib.request import urlopen

from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND, set_vllm_onthefly_hqq_quant
set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)

#VLLM model
from vllm import LLM
from vllm.sampling_params import SamplingParams

#On-the-fly
set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head', 'visual'])
##set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic_int8', skip_modules=['lm_head', 'visual'])
##set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic_fp8', skip_modules=['lm_head', 'visual'])
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.80, dtype=torch.float16,
    max_num_seqs=5,
    mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 1,},
    disable_mm_preprocessor_cache=False,
)

question = 'What is this?'
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/767px-Cat_November_2010-1a.jpg"

img = Image.open(urlopen(image_url)).convert("RGB")

placeholder = "<|image_pad|>"
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
          f"{question}<|im_end|>\n"
          "<|im_start|>assistant\n")


inputs = {"prompt": prompt,"multi_modal_data": {'image': img},}
sampling_params = SamplingParams(temperature=0.2, max_tokens=64, stop_token_ids=None)

outputs = llm.generate(inputs, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
