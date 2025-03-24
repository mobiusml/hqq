#pip install git+https://github.com/huggingface/transformers/
#pip install hqq gemlite

import torch
device        = 'cuda:0'
backend       = "gemlite" 
compute_dtype = torch.bfloat16 
cache_dir     = None
model_id      = 'google/gemma-3-12b-it'
########################################################################################################################################
from transformers import HqqConfig, Gemma3ForConditionalGeneration, AutoProcessor
import gemlite

#Load
processor    = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
quant_config = HqqConfig(nbits=4, group_size=64, axis=1, skip_modules=['lm_head', 'vision_tower'])
model       = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", cache_dir=cache_dir, quantization_config=quant_config, device_map='cuda')

from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend=backend, verbose=True) 

#Gemma3 doesn't support static cache in transformers yet.

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=compute_dtype)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False)[0][input_len:]
    decoded    = processor.decode(generation, skip_special_tokens=True)

print(decoded)

