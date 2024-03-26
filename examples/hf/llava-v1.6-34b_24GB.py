# pip install git+https://github.com/mobiusml/hqq.git
# pip install flash-attn --no-build-isolation
# pip install sentencepiece #for lava-next tokenizer
# num_threads=32; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3 
###################################################################################################
import torch, transformers, os, gc

model_id      = "llava-hf/llava-v1.6-34b-hf"
compute_dtype = torch.float16
device        = 'cuda'
attn_imp      = "flash_attention_2" #flash_attention_2 / sdpa / eager
cache_path    =  '.' 
###################################################################################################
#Load model on CPU
processor = transformers.LlavaNextProcessor.from_pretrained(model_id, use_fast=False)
tokenizer = processor.tokenizer
model     = transformers.LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation=attn_imp)

#Quantize and offload to GPU
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ

############################################################
#Faster and better quality | Runtime VRAM ~25GB
#quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, offload_meta=False) 

#Designed to fit a 24GB    | Runtime VRAM ~23.4GB
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True, offload_meta=True) 
quant_config['scale_quant_params']['group_size'] = 64
quant_config['zero_quant_params']['group_size']  = 64

############################################################
#Quantize the language model
LlamaHQQ.quantize_model(model.language_model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Move the rest of the model
model.vision_tower          = model.vision_tower.to(device=device, dtype=compute_dtype)
model.multi_modal_projector = model.multi_modal_projector.to(device=device, dtype=compute_dtype)
model.image_newline.data    = model.image_newline.data.to(device=device, dtype=compute_dtype)

#Compile for faster processing
#model                      = torch.compile(model)

#Set eval mode
model.generation_config.use_cache = True
model = model.eval();

###################################################################################################
#Generation functions
###################################################################################################
from PIL import Image
import requests
import torch
from threading import Thread

@torch.inference_mode()
def llava_gen(image, prompt, system_prompt="Answer the questions.", assistant_tag="assistant\n", max_new_tokens=256, do_sample=False):
	prompt_raw =  "<|im_start|>system\n" + system_prompt + "<|im_end|><|im_start|>user\n<image>\n" + prompt + "<|im_end|><|im_start|>assistant\n"
	image      = Image.open(image) if isinstance(image, str) else image
	output     = model.generate(**processor(prompt_raw, image, return_tensors='pt').to(device), 
								max_new_tokens=max_new_tokens, 
								do_sample=do_sample,
								pad_token_id=tokenizer.pad_token_id,
								top_p=0.90 if do_sample else None,
								top_k=50 if do_sample else None,
								temperature= 0.6 if do_sample else None,
								num_beams=1,
								repetition_penalty=1.2,
								) 
	response   = processor.batch_decode(output, skip_special_tokens=True)
	response   = response[0].split(assistant_tag)[1].strip()
	return response
	
@torch.inference_mode()
def llava_chat(image, prompt, system_prompt="Answer the questions.", max_new_tokens=256, do_sample=False):

    streamer = transformers.TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    prompt_raw =  "<|im_start|>system\n" + system_prompt + "<|im_end|><|im_start|>user\n<image>\n" + prompt + "<|im_end|><|im_start|>assistant\n"

    generate_params = dict(
        processor(prompt_raw, image, return_tensors="pt").to(device),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        top_p=0.90 if do_sample else None,
        top_k=50 if do_sample else None,
        temperature= 0.6 if do_sample else None,
        num_beams=1,
        repetition_penalty=1.2,
    )

    t = Thread(target=model.generate, kwargs=generate_params)
    t.start()

    print('------------------------------------------------------------')
    
    print("User: ", prompt); 
    print("Assistant: ");
    outputs = ""
    for text in streamer:
        outputs += text
        print(text, end="", flush=True)

    torch.cuda.empty_cache()
  
    return outputs

###################################################################################################
#Generate
###################################################################################################

image = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)

out   = llava_chat(image, prompt="What is shown in this image?", max_new_tokens=256, do_sample=False)
