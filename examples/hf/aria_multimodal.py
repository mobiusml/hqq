#!pip install transformers hqq;
#!pip install flash-attn --no-build-isolation
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 
########################################################################
import torch
device        = 'cuda:0'
backend       = 'torchao_int4' #"torchao_int4" (4-bit only) or "bitblas" (4-bit + 2-bit)
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = '.' 
model_id      = 'rhymes-ai/Aria'

########################################################################
#Load model
from transformers import AutoModelForCausalLM, AutoProcessor
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

#Quantize
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="flash_attention_2", trust_remote_code=True)
quant_config = BaseQuantizeConfig(nbits=4, group_size=128, axis=1)
AutoHQQHFModel.quantize_model(model.language_model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Move the vision model to the device
model.multi_modal_projector = model.multi_modal_projector.to(device)
model.vision_tower          = model.vision_tower.to(device)

#Optimize
from hqq.utils.patching import prepare_for_inference
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)
prepare_for_inference(model.language_model, backend=backend, verbose=True)
########################################################################
from PIL import Image
import requests
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

image = Image.open(requests.get(image_path, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": "what is the image?", "type": "text"},
        ],
    }
]

text   = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output     = model.generate(**inputs, max_new_tokens=500, stop_strings=["<|im_end|>"], tokenizer=processor.tokenizer, do_sample=True, temperature=0.9)
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    result     = processor.decode(output_ids, skip_special_tokens=True)

print(result)
