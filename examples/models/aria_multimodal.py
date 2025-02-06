#pip uninstall torch torchvision;
#pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121;

#flash_attn_file=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl;
#pip install --no-dependencies --upgrade $flash_attn_file;

#pip install --upgrade sentencepiece transformers hqq;
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 

########################################################################################
#Load 
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from hqq.utils.aria import quantize_model, patch_model_for_compiled_runtime, generate

#Load
processor = AutoProcessor.from_pretrained('rhymes-ai/Aria', cache_dir='.', trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('rhymes-ai/Aria', 
                                            cache_dir='.',
                                            torch_dtype=torch.bfloat16, 
                                            trust_remote_code=True, 
                                            attn_implementation={"text_config": "sdpa", "vision_config": "flash_attention_2"})


quantize_model(model)
patch_model_for_compiled_runtime(model, processor, warmup=True)

import time 
t1=time.time()
print(generate(model, processor, img_path="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", prompt="what is the image?"))
t2=time.time()
print(t2-t1) #3x less memory, 3.5-4x faster!