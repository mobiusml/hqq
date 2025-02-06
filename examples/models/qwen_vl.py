# pip install transformers==4.46.3
# pip install qwen-vl-utils
# pip install git+https://github.com/mobiusml/hqq.git;
# pip install bitblas #to use the bitblas backend
########################################################################
import torch
device        = 'cuda:0'
backend       = "torchao_int4" #'torchao_int4' #"torchao_int4" (4-bit only) or "bitblas" (4-bit + 2-bit) or "gemlite" (8-bit, 4-bit, 2-bit, 1-bit)
compute_dtype = torch.bfloat16 if backend=="torchao_int4" else torch.float16
cache_dir     = None #'.' 
model_id      = "Qwen/Qwen2-VL-7B-Instruct"

######################################################################
#Helper functions
from torch.nn.attention import sdpa_kernel, SDPBackend
from hqq.utils.generation_hf import patch_accelerate_device_hook

@torch.inference_mode()
def generate(model, processor, img_path, prompt, max_new_tokens=128, do_sample=True):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=do_sample, temperature=0.9, cache_implementation="static")
    output_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    result     = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return result

def patch_model_for_compiled_runtime(model, processor, compile=True, warmup=True, patch_accelerate=True):
    if patch_accelerate:
        patch_accelerate_device_hook()

    torch._dynamo.config.inline_inbuilt_nn_modules = False  # torch 2.5.0 fix
    torch._dynamo.config.capture_scalar_outputs = True

    model.config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.eval()

    if(compile):
        forward_compiled = torch.compile(model.language_model.forward, mode="reduce-overhead", fullgraph=True)
    else:
        forward_compiled = model.language_model.forward
    forward_simple   = model.language_model.forward

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})

    def custom_forward(*args, **kwargs):
        if (
            (len(args) > 0 and args[0].shape[-1] == 1)
            or ("inputs_embeds" in kwargs and kwargs["inputs_embeds"].shape[1] == 1)
        ):  
            with sdpa_kernel([SDPBackend.MATH]):
                out = forward_compiled(*args, **kwargs)
            return out

        out = forward_simple(*args, **kwargs)
        return out

    model.language_model.forward = custom_forward

    if warmup:
        for _ in range(3):
            generate(
                model,
                processor,
                img_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                prompt="Describe this image.",
            )

######################################################################
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model     = Qwen2VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=compute_dtype, attn_implementation="sdpa").eval()
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, max_pixels=1280 * 28 * 28)

########################################################################
#Quantize and offload to GPU
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference

quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1) 
model.language_model = model.model
AutoHQQHFModel.quantize_model(model.language_model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
model.visual  = model.visual.to(device=device, dtype=compute_dtype)
model.lm_head = model.lm_head.to(device=device, dtype=compute_dtype)

#Patch
prepare_for_inference(model.language_model, backend=backend, verbose=True)

#Patch
patch_model_for_compiled_runtime(model, processor, warmup=True, patch_accelerate=True, compile=False) #compile=True breaks :(
########################################################################

out = generate(model,
               processor,
               img_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
               prompt="Describe this image.")
