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
  4-bit models with `axis=1` can use optimized inference fused kernels like torchao's int4_gemm. This is the same kernel used in <a href="https://github.com/pytorch-labs/gpt-fast">gpt-fast</a> and based on our benchmarks, it's the fastest kernel available right now. We also support the <a href="https://github.com/IST-DASLab/marlin/tree/master/marlin">Marlin</a> kernel. Moreover, we focus on making hqq fully compatible with `torch.compile` which speeds-up both training and inference. For more details, please refer to the backend section below. <br>

  <b>What quantization settings should I use?</b><br>
  You should start with `nbits=4, group_size=64, axis=1`. These settings offer a good balance between quality, vram usage and speed. If you want better results with the same vram usage, switch to `axis=0` and use the ATEN backend. If you want to use lower like `nbits=2`, you should use `axis=0`with a low group-size via HQQ+, meaning adding low-rank adapters and fine-tune with a small dataset. <br>
	
  <b>What does the `axis` parameter mean? </b><br>
  The `axis` parameter is the axis along which grouping is performed. In general `axis=0` gives better results than `axis=1`, especially at lower bits. However, the optimized inference runtime only supports `axis=1` for the moment.<br>
	
  <b>What is the difference between HQQ and HQQ+?</b><br>
  HQQ+ is HQQ with trainable low-rank adapters to improve the quantization quality at lower bits.<br>

</details>

### Installation 
First, make sure you have a Pytorch 2 version that matches your CUDA version: https://pytorch.org/

You can install hqq via  ```pip install hqq```. 

To get the latest version, you can install the core library directly via ```pip install git+https://github.com/mobiusml/hqq.git```. 

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
- ```quant_zero``` (bool): if True, it quantizes the zero-point to 8-bit without grouping.
- ```quant_scale``` (bool): if True, it quantizes the scaling factor to 8-bit with a group_size of 128.
- ```offload_meta``` (bool): if True, meta-data is offloaded to the CPU.
- ```view_as_float``` (bool): if True, the quantized parameter is viewed as float instead of a int type.

Setting ```offload_meta=True``` drastically decreases the GPU memory requirements but makes processing slower for smaller group-sizes. When turned on, you can run Llama2-70B and Mixtral with HQQ 2-bit using only 18.8GB and 13GB VRAM respectively.

### Backend
#### Native Backends
The following native backends can be used by the `HQQLinear` module:
```Python
HQQLinear.set_backend(HQQBackend.PYTORCH)          #Pytorch backend
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)  #Compiled Pytorch
HQQLinear.set_backend(HQQBackend.ATEN)             #Aten/CUDA backend
```
The ```HQQBackend.ATEN``` backend is automatically installed and used by default when available.
Note that ```HQQBackend.ATEN```  only supports `axis=0`. For `axis=1` you need to use ```HQQBackend.PYTORCH``` or ```HQQBackend.PYTORCH_COMPILE```.

Below you can find the speed-up benchmark with various backends, ```HQQBackend.PYTORCH``` being the baseline:

 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama27b_titanrtx.png" alt="Titan RTX" style="width:48%">
	<img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama270b_a100.png" alt="A100" style="width:48%">
  </div>
 </center>
</div> 

#### Faster Inference
We support external backends for faster inference with fused kernels. You can enable one of the backends after the model was quantized as follows:
```Python
from hqq.utils.patching import prepare_for_inference

#Pytorch backend that makes the model compatible with fullgrah torch.compile: works with any settings
#prepare_for_inference(model) 

#Torchao's tiny_gemm backned (fastest): nbits=4, compute_dtype=bfloat16, axis=1
prepare_for_inference(model, backend="torchao_int4") 

#Marlin backend: nbits=4, axis=1, compute_dtype=float16, group_size=None
#prepare_for_inference(model, backend="marlin", allow_merge=True) 
```
These backends only work with 4-bit quantization and `axis=1`. Additionally, for <a href="https://github.com/IST-DASLab/marlin.git">Marlin</a>, we only support `group_size=None`. Below you can find a comparison between the different backends. The torchao kernel reaches 195 tokens/sec (generation speed) on a 4090.

<p align="center">
    <img src="https://github.com/mobiusml/hqq/blob/master/imgs/llama_int4_4090.png" alt="backend 4090" >
</p>


### Usage with Models
#### Transformers 🤗
For usage with HF's transformers, see the example below from the <a href="https://huggingface.co/docs/transformers/main/en/quantization#hqq">documentation</a>:
```Python
from transformers import AutoModelForCausalLM, HqqConfig

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, axis=1)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```
<b>Note</b>: You can't save/load quantized models directly via `save_pretrained` with this approach. Use the save/load calls from the hqq lib instead.

#### HQQ Lib
You can also utilize the HQQ library to quantize transformers models:
```Python
#Load the model on CPU
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)

#Quantize
from hqq.models.hf.base import AutoHQQHFModel
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1) 
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
```
#### Save/Load
You can save/load quantized models as follows:
```Python
from hqq.models.hf.base import AutoHQQHFModel

#Save: Make sure to save the model BEFORE any patching
AutoHQQHFModel.save_quantized(model, save_dir)

#Load
model = AutoHQQHFModel.from_quantized(save_dir)
```
#### Setting a backend
You can set a native backned as follows:
```Python
HQQLinear.set_backend(HQQBackend.ATEN if axis==0 else HQQBackend.PYTORCH_COMPILE)
```

You can patch for faster inference as explained in the <a href="https://github.com/mobiusml/hqq/edit/master/Readme.md#backend">backend</a> section:
```Python
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="torchao_int4")
```

#### Custom HF Models
`AutoHQQHFModel` is meant to be compatible with any transformers model. However, its adaptability comes with a drawback - it may encounter issues or experience sluggishness when processing layers. If you encounter such problems, you have the option to create a custom model with clearly defined patching logic to replace `AutoHQQHFModel`. Below are examples of popular models you can utilize or expand upon for this purpose:

```Python
from hqq.models.hf.llama import LlamaHQQ #Llama
from hqq.models.hf.mistral import MistralHQQ #Mistral
from hqq.models.hf.mixtral import MixtralHQQ #Mixtral
```

### Custom Quantization Configurations ⚙️
You can set up various quantization configurations for different layers by specifying the settings for each layer name:
#### Transformers 🤗
```Python
# Each linear layer with the same tag will use a dedicated quantization config
q4_config = {'nbits':4, 'group_size':64, 'quant_zero':False, 'quant_scale':False}
q3_config = {'nbits':3, 'group_size':32, 'quant_zero':False, 'quant_scale':False}

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
q4_config    = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False) 
q3_config    = BaseQuantizeConfig(nbits=3, group_size=32, quant_zero=False, quant_scale=False)

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
You can use HQQ for LoRA training as follows:
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

We provide a complete example to train a model with HQQ/LoRA that you can find in ```examples/lora/train_hqq_lora_example.py```.

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
