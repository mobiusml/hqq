################################################################################################
# git clone https://github.com/mobiusml/hqq/; cd hqq; pip install .;
# cd hqq/kernels && python setup_cuda.py install; cd ..; cd ..; cd ..;
# pip install flash-attn --no-build-isolation
# pip install transformers --upgrade
# num_threads=8; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython 
################################################################################################

import torch
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
model_id   = "mistralai/Mixtral-8x7B-Instruct-v0.1"
cache_path = '.'
model      = HQQModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
tokenizer  = AutoTokenizer.from_pretrained(model_id,       cache_dir=cache_path) 

#Quantize params
from hqq.core.quantize import *
attn_prams     = BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=True) 
experts_params = BaseQuantizeConfig(nbits=2, group_size=16, offload_meta=True) 
attn_prams['scale_quant_params']['group_size'] = 256
attn_prams['zero_quant_params']['group_size']  = 256

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
model.quantize_model(quant_config=quant_config, compute_dtype=torch.float16)
model.eval();

################################################################################################
#Set backend
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
model = torch.compile(model)

#Warmup 
for i in range(10):
    with torch.no_grad():
        out = model(torch.ones((1, 1024), dtype=torch.int32, device='cuda'))
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
    outputs = []
    for text in streamer:
        outputs.append(text)
        print(text, end="", flush=True)

    return outputs

################################################################################################
#Generation
outputs = chat_processor("How do I build a car?", max_new_tokens=1000, do_sample=False)
