## Half-Quadratic Quantization (HQQ)
This repository contains the official implementation of Half-Quadratic Quantization (<b>HQQ</b>) presented in our articles: 
* HQQ: https://mobiusml.github.io/hqq_blog/
* HQQ+: https://mobiusml.github.io/1bit_blog/

### What is HQQ?
<b>HQQ</b> is a fast and accurate model quantizer that skips the need for calibration data. Quantize the largest models, without calibration data, in just a few minutes at most 🚀.

<details>
  <summary>FAQ </summary>
 <b> Why should I use HQQ instead of other quantization methods? </b><br>
<ul>
<li> HQQ is very fast to quantize models.</li>
<li> It supports 8,4,3,2,1 bits.</li>
<li> You can use it on any model (LLMs, Vision, etc.).</li>
<li> The dequantization step is a linear operation, this means that HQQ is compatbile with various optimized CUDA/Triton kernels.</li>
<li> HQQ is compatible with peft training.</li>
<li> We try to make HQQ fully compatible `torch.compile` for faster inference and training.</li>
</ul>
  
  <b>What is the quality of the quantized models? </b><br>
  We have detailed benchmarks on both language and vision models. Please refer to our blog posts: <a href="https://mobiusml.github.io/hqq_blog/">HQQ</a>, <a href="https://mobiusml.github.io/1bit_blog/">HQQ+</a>.<br> 

  <b>What is the speed of the quantized models?</b><br>
  4-bit models with `axis=1` can use optimized inference fused kernels. Moreover, we focus on making hqq fully compatible with `torch.compile` which speeds-up both training and inference. For more details, please refer to the backend section below. <br>

  <b>What quantization settings should I use?</b><br>
  You should start with `nbits=4, group_size=64, axis=1`. These settings offer a good balance between quality, vram usage and speed. If you want better results with the same vram usage, switch to `axis=0` and use the ATEN backend, but this setting is not supported for fast inference. <br>
  
  <b>What does the `axis` parameter mean? </b><br>
  The `axis` parameter is the axis along which grouping is performed. In general `axis=0` gives better results than `axis=1`, especially at lower bits. However, the optimized inference runtime only supports `axis=1` for the moment.<br>
  
  <b>What is the difference between HQQ and HQQ+?</b><br>
  HQQ+ is HQQ with trainable low-rank adapters to improve the quantization quality at lower bits.<br>

</details>

### Installation 
First, make sure you have a Pytorch 2 version that matches your CUDA version: https://pytorch.org/

You can install hqq via  
```
#latest stable version
pip install hqq;

#Latest updates - recommended
pip install git+https://github.com/mobiusml/hqq.git; 
```

Alternatively, clone the repo and run ```pip install .``` from this current folder. 

### Basic Usage
To perform quantization with HQQ, you simply need to replace the linear layers ( ```torch.nn.Linear```) as follows:
```Python
from hqq.core.quantize import *
#Quantization settings
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)

#Replace your linear layer 
hqq_layer = HQQLinear(your_linear_layer, #torch.nn.Linear or None 
                      quant_config=quant_config, #quantization configuration
                      compute_dtype=torch.float16, #compute dtype
                      device='cuda', #cuda device
                      initialize=True, #Use False to quantize later
                      del_orig=True #if True, delete the original layer
                      )
```

The quantization parameters are set as follows:

- ```nbits``` (int): supports 8, 4, 3, 2, 1 bits.
- ```group_size``` (int): no restrictions as long as ```weight.numel()``` is divisible by the ```group_size```.
- ```view_as_float``` (bool): if True, the quantized parameter is viewed as float instead of an int type.

### Usage with Models
#### Transformers 🤗
For usage with HF's transformers, see the example below from the <a href="https://huggingface.co/docs/transformers/main/en/quantization#hqq">documentation</a>:
```Python
from transformers import AutoModelForCausalLM, HqqConfig

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=4, group_size=64)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```
You can save/load quantized models as regular transformers models via `save_pretrained` / `from_pretrained`.

#### HQQ Lib
You can also utilize the HQQ library to quantize transformers models:
```Python
#Load the model on CPU
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)

#Quantize
from hqq.models.hf.base import AutoHQQHFModel
quant_config = BaseQuantizeConfig(nbits=4, group_size=64) 
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
```
You can save/load quantized models as follows:
```Python
from hqq.models.hf.base import AutoHQQHFModel

#Save: Make sure to save the model BEFORE any patching
AutoHQQHFModel.save_quantized(model, save_dir)

#Load
model = AutoHQQHFModel.from_quantized(save_dir)
```

❗ Note that models saved via the hqq lib are not compatible with `.from_pretrained()`

### Backends
#### Native Backends
The following native dequantization backends can be used by the `HQQLinear` module:
```Python
HQQLinear.set_backend(HQQBackend.PYTORCH)          #Pytorch backend - Default
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)  #Compiled Pytorch
HQQLinear.set_backend(HQQBackend.ATEN)             #Aten/CUDA backend - only axis=0 supported
```
❗ Note that ```HQQBackend.ATEN```  only supports `axis=0`. 

#### Optimized Inference
We support external backends for faster inference with fused kernels. You can enable one of the backends after the model was quantized as follows:
```Python
from hqq.utils.patching import prepare_for_inference

#Pytorch backend that makes the model compatible with fullgrah torch.compile: works with any settings
#prepare_for_inference(model) 

#Torchao's tiny_gemm backned (fastest): nbits=4, compute_dtype=bfloat16, axis=1
prepare_for_inference(model, backend="torchao_int4") 

#Gemlite backend: nbits=4/2/1, compute_dtype=float16, axis=1
#prepare_for_inference(model, backend="gemlite") 

#Bitblas backend: nbits=4/2, compute_dtype=float16, axis=1
#prepare_for_inference(model, backend="bitblas") 
```
Note that these backends only work with `axis=1`. Additional restrictions apply regarding the group-size values depending on the backend. You should expect ~158 tokens/sec with a Llama3-8B 4-bit quantized model on a 4090 RTX.

When a quantization config is not supported by the specified inference backend, hqq will fallback to the native backend. 

### Custom Quantization Configurations ⚙️
You can set up various quantization configurations for different layers by specifying the settings for each layer name:
#### Transformers 🤗
```Python
# Each linear layer with the same tag will use a dedicated quantization config
q4_config = {'nbits':4, 'group_size':64}
q3_config = {'nbits':3, 'group_size':32}

quant_config  = HqqConfig(dynamic_config={
  'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
})
```
#### HQQ lib
```Python
from hqq.core.quantize import *
q4_config    = BaseQuantizeConfig(nbits=4, group_size=64) 
q3_config    = BaseQuantizeConfig(nbits=3, group_size=32)

quant_config = {'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
}
```

### Peft Training
Peft training is directly supported in the HuggingFace's <a href="https://huggingface.co/docs/peft/v0.12.0/en/developer_guides/quantization#hqq-quantization"> peft library</a>. If you still want to use hqq-lib's peft utilities, here's how: 

```Python
#First, quantize/load a quantized HQQ model the
from hqq.core.peft import PeftUtils

base_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':64, 'dropout':0.05, 'train_dtype':torch.float32}
lora_params      = {'self_attn.q_proj': base_lora_params,
                    'self_attn.k_proj': base_lora_params,
                    'self_attn.v_proj': base_lora_params,
                    'self_attn.o_proj': base_lora_params,
                    'mlp.gate_proj'   : None,
                    'mlp.up_proj'     : None,
                    'mlp.down_proj'   : None}


#Add LoRA to linear/HQQ modules
PeftUtils.add_lora(model, lora_params)

#Optional: set your backend
HQQLinear.set_backend(HQQBackend.ATEN if axis==0 else HQQBackend.PYTORCH_COMPILE)

#Train ....

#Convert LoRA weights to the same model dtype for faster inference
model.eval()
PeftUtils.cast_lora_weights(model, dtype=compute_dtype)

#Save LoRA weights
PeftUtils.save_lora_weights(model, filename)

#Load LoRA weights: automatically calls add_lora 
PeftUtils.load_lora_weights(model, filename)
```

We provide a complete example to train a model with HQQ/LoRA that you can find in ```examples/hqq_plus.py```.

If you want to use muti-gpu training via FSDP, check out this awesome repo by Answer.AI: https://github.com/AnswerDotAI/fsdp_qlora

### Examples 
We provide a variety of examples demonstrating model quantization across different backends within the ```examples```  directory.

### Citation 📜
```
@misc{badri2023hqq,
title  = {Half-Quadratic Quantization of Large Machine Learning Models},
url    = {https://mobiusml.github.io/hqq_blog/},
author = {Hicham Badri and Appu Shaji},
month  = {November},
year   = {2023}
```
