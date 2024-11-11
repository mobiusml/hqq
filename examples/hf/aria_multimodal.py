#pip uninstall torch torchvision;
#pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121;

# flash_attn_file=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl;
# pip install --no-dependencies --upgrade $flash_attn_file;

#pip install --upgrade sentencepiece transformers hqq;
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 

########################################################################################
import torch, gc
device        = 'cuda:0'
backend       = 'torchao_int4'
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = '.' 
model_id      = 'rhymes-ai/Aria'

########################################################################################
#Load model
from transformers import AutoModelForCausalLM, AutoProcessor
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

#Load
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, device_map=device, attn_implementation="flash_attention_2", trust_remote_code=True)

#Quantize
#model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="flash_attention_2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="sdpa", trust_remote_code=True)

#model = model.cuda()

########################################################################################
attn_quant_config    = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
experts_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)

def autoname_modules(model) -> None:
    for name, module in model.named_modules():
        module.name = name

from torch import Tensor
def int4mm(x: Tensor, weight_int4pack: Tensor, scales_and_zeros: Tensor, groupsize: int, out_features: int) -> Tensor:
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c

# @torch.library.custom_op("aria::forward_seq_gemm", mutates_args=())
# def forward_seq_gemm(input: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int, out_features: int, name: str) -> torch.Tensor: 
#     num_tokens = input.shape[0]
#     output     = torch.empty(num_tokens, out_features, dtype=input.dtype, device=input.device)

#     cumsum_num_tokens     = torch.zeros(len(tokens_per_expert) + 1, dtype=torch.long, device=input.device)
#     cumsum_num_tokens[1:] = torch.cumsum(tokens_per_expert, dim=0)
    
#     selected_experts      = torch.where(torch.diff(cumsum_num_tokens).view(-1, 1))[0]

#     _weights   = HQQGroupedGemm.CACHE[name]['weights'][selected_experts] 
#     _scales    = HQQGroupedGemm.CACHE[name]['scales_and_zeros'][selected_experts]
#     group_size = HQQGroupedGemm.CACHE[name]['group_size']

#     num_active_experts = len(selected_experts)

#     for i in range(num_active_experts):
#         expert_num = selected_experts[i]
#         start      = cumsum_num_tokens[expert_num]
#         end        = cumsum_num_tokens[expert_num + 1]
#         output[start:end] = int4mm(input[start:end], _weights[i], _scales[i], group_size, out_features)
    
#     return output

###############################
def forward_seq_gemm_prefill(input: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int, out_features: int, name: str) -> torch.Tensor: 
    num_tokens = input.shape[0]
    output     = torch.empty(num_tokens, out_features, dtype=input.dtype, device=input.device)

    cumsum_num_tokens     = torch.zeros(len(tokens_per_expert) + 1, dtype=torch.long, device=input.device)
    cumsum_num_tokens[1:] = torch.cumsum(tokens_per_expert, dim=0)
    
    #selected_experts = torch.where(torch.diff(cumsum_num_tokens).view(-1, 1))[0]
    #num_active_experts = len(selected_experts)

    num_active_experts = (tokens_per_expert > 0).sum().int() #this should be static
    selected_experts = torch.topk(torch.diff(cumsum_num_tokens), k=num_active_experts, dim=0, largest=True)[1].sort()[0]

    _weights    = HQQGroupedGemm.CACHE[name]['weights'][selected_experts] 
    _scales     = HQQGroupedGemm.CACHE[name]['scales_and_zeros'][selected_experts]
    _group_size = HQQGroupedGemm.CACHE[name]['group_size']

    _start = cumsum_num_tokens[selected_experts]
    _end   = cumsum_num_tokens[selected_experts + 1]
    
    for i in range(num_active_experts):
        start = _start[i] 
        end   = _end[i] 
        output[start:end] = int4mm(input[start:end], _weights[i], _scales[i], _group_size, out_features)

    return output

def forward_seq_gemm_decode(input: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int, out_features: int, name: str) -> torch.Tensor: 
    num_tokens = input.shape[0]
    output     = torch.empty(num_tokens, out_features, dtype=input.dtype, device=input.device)

    cumsum_num_tokens     = torch.zeros(len(tokens_per_expert) + 1, dtype=torch.long, device=input.device)
    cumsum_num_tokens[1:] = torch.cumsum(tokens_per_expert, dim=0)
    
    #selected_experts = torch.where(torch.diff(cumsum_num_tokens).view(-1, 1))[0]
    #num_active_experts = len(selected_experts)

    num_active_experts = 6
    selected_experts = torch.topk(torch.diff(cumsum_num_tokens), k=num_active_experts, dim=0, largest=True)[1].sort()[0]

    _weights    = HQQGroupedGemm.CACHE[name]['weights'][selected_experts] 
    _scales     = HQQGroupedGemm.CACHE[name]['scales_and_zeros'][selected_experts]
    _group_size = HQQGroupedGemm.CACHE[name]['group_size']

    _start = cumsum_num_tokens[selected_experts]
    _end   = cumsum_num_tokens[selected_experts + 1]
    
    for i in range(num_active_experts):
        start = i
        end   = i + 1 
        output[start:end] = int4mm(input[start:end], _weights[i], _scales[i], _group_size, out_features)

    return output

@torch.library.custom_op("aria::forward_seq_gemm", mutates_args=())
def forward_seq_gemm(input: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int, out_features: int, name: str) -> torch.Tensor: 
    num_tokens = input.shape[0]
    if(num_tokens > 6):
        return forward_seq_gemm_prefill(input, tokens_per_expert, num_experts, out_features, name)
    else:
        return forward_seq_gemm_decode(input, tokens_per_expert, num_experts, out_features, name)


@torch.library.register_fake("aria::forward_seq_gemm")
def forward_seq_gemm_fake(input: torch.Tensor, tokens_per_expert: torch.Tensor, num_experts: int, out_features: int, name: str) -> torch.Tensor: 
    return torch.empty(input.shape[0], out_features, dtype=input.dtype, device=input.device)
###############################3

from functorch.experimental.control_flow import cond
from hqq.utils.patching import patch_hqq_to_aoint4, patch_hqq_inference
class HQQGroupedGemm(torch.nn.Module):
    CACHE = {}
    def __init__(self, grouped_gemm_layer, quant_config, backend):
        super().__init__()

        self.quant_expert(grouped_gemm_layer, quant_config, backend)
        self.quant_config = quant_config

        self.name = grouped_gemm_layer.name
        HQQGroupedGemm.CACHE[self.name] = {'weights': self.weights, 'scales_and_zeros': self.scales_and_zeros, 'group_size':self.group_size}

    def quant_expert(self, mlp_layer, quant_config, backend='torchao_int4'):
        weight = mlp_layer.weight
        num_experts, in_features, out_features = weight.shape
        W_nbits = quant_config['weight_quant_params']['nbits']
        gs      = quant_config['weight_quant_params']['group_size']

        weight_int4pack, scales_and_zeros = [], []
        for j in range(num_experts):      
            hqq_layer = patch_hqq_to_aoint4(HQQLinear.from_weights(weight=weight[j].T, bias=None, quant_config=quant_config, compute_dtype=compute_dtype, device=device, del_orig=True), None)
            weight_int4pack.append(hqq_layer.weight_int4pack[None,:])
            scales_and_zeros.append(hqq_layer.scales_and_zeros[None,:])

        self.weights          = torch.cat(weight_int4pack)
        self.scales_and_zeros = torch.cat(scales_and_zeros)
        self.in_features      = in_features
        self.out_features     = out_features
        self.num_experts      = num_experts
        self.group_size       = quant_config['weight_quant_params']['group_size']

        del hqq_layer
        torch.cuda.empty_cache()

    def forward(self, input, tokens_per_expert): #sequential_gemm
        return forward_seq_gemm(input, tokens_per_expert, self.num_experts, self.out_features, self.name)

from tqdm import tqdm

#We first name the modules to keep track of the layers in HQQGroupedGemm.CACHE
autoname_modules(model.language_model)

for i in tqdm(range(len(model.language_model.model.layers))):
    model.language_model.model.layers[i].mlp.experts.fc1 = HQQGroupedGemm(model.language_model.model.layers[i].mlp.experts.fc1, quant_config=experts_quant_config, backend=backend)
    model.language_model.model.layers[i].mlp.experts.fc2 = HQQGroupedGemm(model.language_model.model.layers[i].mlp.experts.fc2, quant_config=experts_quant_config, backend=backend)
gc.collect()

#Quantize the rest
AutoHQQHFModel.quantize_model(model.language_model, quant_config=attn_quant_config, compute_dtype=compute_dtype, device=device)

#Remove losses
# import moe_llm
# moe_llm.apply_z_loss   = lambda logits: logits
# moe_llm.apply_aux_loss = lambda logits, tokens_per_expert, scores: scores

#Move the vision model to the device
model.multi_modal_projector = model.multi_modal_projector.to(device)
model.vision_tower          = model.vision_tower.to(device)

#Optimize
from hqq.utils.patching import prepare_for_inference
HQQLinear.set_backend(HQQBackend.ATEN if experts_quant_config['weight_quant_params']['axis'] == 0 else HQQBackend.PYTORCH_COMPILE)
prepare_for_inference(model.language_model, backend=backend, verbose=True)

########################################################################################
import requests
from PIL import Image

def generate(img_path, prompt):

    image = Image.open(requests.get(img_path, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    text   = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output     = model.generate(**inputs, max_new_tokens=500, stop_strings=["<|im_end|>"], 
                                    tokenizer=processor.tokenizer, do_sample=True, temperature=0.9, 
                                    cache_implementation="static"
                                    )
        output_ids = output[0][inputs["input_ids"].shape[1]:]
        result     = processor.decode(output_ids, skip_special_tokens=True)

    return result

####################################################################################
# from torch.nn.attention import sdpa_kernel, SDPBackend
# _model = model.language_model
# _tokenizer = processor.tokenizer

# _model.config.use_cache = True
# _model.generation_config.cache_implementation = "static"
# _model.eval()

# inputz = torch.randint(0, 100, (1, 1), dtype=torch.int32, device='cuda')
# with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
#     out = _model.forward(inputz)

# print('---------------------------------------------------------------------------')
# # torch._dynamo.config.capture_dynamic_output_shape_ops = True
# # torch._dynamo.config.capture_scalar_outputs = True

# forward_compiled = torch.compile(_model.forward, mode="reduce-overhead", fullgraph=True)
# forward_simple = model.forward

# with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
#     out = forward_compiled(inputz)

# with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
#     out = forward_compiled(inputz)

########################################################################################

#Torch.compile
from hqq.utils.generation_hf import WARMUP_PROMPTS, patch_accelerate_device_hook
from torch.nn.attention import sdpa_kernel, SDPBackend

with torch.no_grad():
    _ = model.language_model(torch.randint(0, 100, (1, 1), dtype=torch.int32, device='cuda'))

torch._dynamo.config.inline_inbuilt_nn_modules = False #torch 2.5.0 fix
def patch_model_for_compiled_runtime(
    model, tokenizer, warmup=True, max_new_tokens=500, patch_accelerate=True
):  
    if(patch_accelerate):
        patch_accelerate_device_hook()

    model.config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.eval();

    forward_compiled = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    forward_simple   = model.forward

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})

    def custom_forward(*args, **kwargs):
        # Prefill phase
        out_fct = forward_simple

        # Decoding pahse
        if (len(args) > 0 and args[0].shape[-1] == 1) or (
            "input_ids"     in kwargs and kwargs["input_ids"].shape[-1] == 1) or (
             "inputs_embeds" in kwargs and kwargs["inputs_embeds"].shape[1] == 1):
            out_fct = forward_compiled
            #print('Decoding')

        with sdpa_kernel([SDPBackend.MATH]):
            out = out_fct(*args, **kwargs)
        
        return out

    model.forward = custom_forward


patch_model_for_compiled_runtime(model.language_model, processor.tokenizer)

# #Try
# with torch.no_grad():
#     _ = model.language_model(torch.randint(0, 100, (1, 1), dtype=torch.int32, device='cuda'))

# #model.multi_modal_projector.forward = torch.compile(model.multi_modal_projector.forward, mode='reduce-overhead', fullgraph=True)
# #model.vision_tower.forward          = torch.compile(model.vision_tower.forward, mode='reduce-overhead', fullgraph=True)

#Warm-up
from tqdm import tqdm
with torch.no_grad():
    for _ in tqdm(range(5)):
        generate(img_path="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", prompt="what is the image?");
gc.collect()
########################################################################################

import time 
t1=time.time()
print(generate(img_path="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", prompt="what is the image?"))
t2=time.time()
print(t2-t1) #A100- VRAM: 49GB - ~6.442729949951172 -> 1.8550639152526855 : ~4x faster
