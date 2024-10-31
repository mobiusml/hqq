#pip install torch torchvision --upgrade;

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

#Quantize
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="flash_attention_2", trust_remote_code=True)
attn_quant_config    = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
experts_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)

from hqq.utils.patching import patch_hqq_to_aoint4, patch_hqq_inference
class HQQGroupedGemm(torch.nn.Module):
    def __init__(self, grouped_gemm_layer, quant_config, backend):
        super().__init__()

        self.hqq_layers   = self.quant_expert(grouped_gemm_layer, quant_config, backend)
        self.in_features  = self.hqq_layers[0].in_features
        self.out_features = self.hqq_layers[0].out_features
        self.num_experts  = len(self.hqq_layers)
        self.quant_config = quant_config

    def quant_expert(self, mlp_layer, quant_config, backend='torchao_int4'):
        weight = mlp_layer.weight
        num_experts, in_features, out_features = weight.shape
        hqq_layers = [None] * num_experts
        W_nbits = quant_config['weight_quant_params']['nbits']
        gs      = quant_config['weight_quant_params']['group_size']
        for j in range(num_experts):
            hqq_layers[j] = HQQLinear.from_weights(weight=weight[j].T, bias=None, quant_config=quant_config, compute_dtype=compute_dtype, device=device, del_orig=True)
            hqq_layers[j] = patch_hqq_inference(hqq_layers[j], None)
            if(backend == 'torchao_int4' and W_nbits == 4 and (gs in [32, 64, 128, 256])):
                hqq_layers[j] = patch_hqq_to_aoint4(hqq_layers[j], None)
        return hqq_layers

    def forward(self, input, tokens_per_expert): #sequential_gemm
        num_tokens        = input.shape[0]
        output            = torch.zeros(num_tokens, self.out_features, dtype=input.dtype, device=input.device)
        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        zero_tensor       = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

        for expert_num in range(self.num_experts):
            start  = cumsum_num_tokens[expert_num]
            end    = cumsum_num_tokens[expert_num + 1]
            if(start == end): continue

            output[start:end] = self.hqq_layers[expert_num](input[start:end])

        return output

from tqdm import tqdm
for i in tqdm(range(len(model.language_model.model.layers))):
    model.language_model.model.layers[i].mlp.experts.fc1 = HQQGroupedGemm(model.language_model.model.layers[i].mlp.experts.fc1, quant_config=experts_quant_config, backend=backend)
    model.language_model.model.layers[i].mlp.experts.fc2 = HQQGroupedGemm(model.language_model.model.layers[i].mlp.experts.fc2, quant_config=experts_quant_config, backend=backend)
gc.collect()

#Quantize the rest
AutoHQQHFModel.quantize_model(model.language_model, quant_config=attn_quant_config, compute_dtype=compute_dtype, device=device)

#Move the vision model to the device
model.multi_modal_projector = model.multi_modal_projector.to(device)
model.vision_tower          = model.vision_tower.to(device)

#Optimize
from hqq.utils.patching import prepare_for_inference
HQQLinear.set_backend(HQQBackend.ATEN if experts_quant_config['weight_quant_params']['axis'] == 0 else HQQBackend.PYTORCH_COMPILE)
prepare_for_inference(model.language_model, backend=backend, verbose=True)
########################################################################################
#Test language model
# with torch.no_grad():
#     _ = model(torch.randint(0, 1000, (1,1), device=device, dtype=torch.int32))


#TODO: add torch.compile
########################################################################################
import requests
from PIL import Image

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
