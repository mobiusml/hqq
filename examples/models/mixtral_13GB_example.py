################################################################################################
# pip install hqq
# pip install flash-attn --no-build-isolation
# pip install transformers --upgrade
# num_threads=8; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython 
################################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
model_id   = "mistralai/Mixtral-8x7B-Instruct-v0.1"
cache_path = '.'
compute_dtype = torch.bfloat16
model      = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="flash_attention_2")
tokenizer  = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path) 

#Quantize params
from hqq.core.quantize import *

attn_prams     = BaseQuantizeConfig(nbits=4, group_size=128, axis=1) 
experts_params = BaseQuantizeConfig(nbits=3, group_size=64, axis=0) 

quant_config = {}
#Attention
quant_config['self_attn.q_proj'] = attn_prams
quant_config['self_attn.k_proj'] = attn_prams
quant_config['self_attn.v_proj'] = attn_prams
quant_config['self_attn.o_proj'] = attn_prams
#Experts
quant_config['block_sparse_moe.experts.w1'] = experts_params
quant_config['block_sparse_moe.experts.w2'] = experts_params
quant_config['block_sparse_moe.experts.w3'] = experts_params

#Quantize
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
model.eval();

################################################################################################
#Set backend
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="torchao_int4", verbose=True)

HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)

#Warmup 
for i in range(10):
    with torch.no_grad():
        out = model(torch.ones((1, 1), dtype=torch.int32, device='cuda'))
del out 
cleanup()

################################################################################################
import transformers 
from threading import Thread

def chat_processor(chat, max_new_tokens=100, do_sample=True):
    tokenizer.use_default_system_prompt = False
    streamer = transformers.TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_params = dict(
        tokenizer("<s> [INST] " + chat + " [/INST] ", return_tensors="pt").to('cuda'),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=0.90,
        top_k=50,
        temperature= 0.6,
        num_beams=1,
        repetition_penalty=1.2,
    )

    t = Thread(target=model.generate, kwargs=generate_params)
    t.start()

    print('------------------------------------------------------------')
    cleanup()
    print(chat); print();
    outputs = []
    for text in streamer:
        outputs.append(text)
        print(text, end="", flush=True)

    return outputs

################################################################################################
#Generation
outputs = chat_processor("How do I build a car?", max_new_tokens=1000, do_sample=False)
