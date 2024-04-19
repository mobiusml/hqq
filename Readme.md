## Half-Quadratic Quantization (HQQ)
This folder contains the code to perform Half-Quadratic Quantization (<b>HQQ</b>) presented in our articles: 
* HQQ: https://mobiusml.github.io/hqq_blog/
* HQQ+: https://mobiusml.github.io/1bit_blog/

### WHat is HQQ?
<b>HQQ</b> is a fast and accurate model quantizer that skips the need for calibration data. It's super simple to implement (just a few lines of code for the optimizer). It can crunch through quantizing the Llama2-70B model in only 4 minutes! 🚀

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

Setting ```offload_meta=True``` drastically decreases the GPU memory requirements but makes processing slightly slower for smaller group-sizes. With this setting, you can run Llama2-70B and Mixtral with HQQ 2-bit using only 18.8GB and 13GB VRAM respectively!

### Backend
You can try to change the backend which could speed-up the runtime:
```Python
HQQLinear.set_backend(HQQBackend.PYTORCH)          #Pytorch backend
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)  #Compiled Pytorch via dynamo
HQQLinear.set_backend(HQQBackend.ATEN)             #C++ Aten/CUDA backend (set automatically by default if available)
```
The ```HQQBackend.ATEN``` backend is automatically installed and used by default when available.

Below you can find the speed-up benchmark with various backends, ```HQQBackend.PYTORCH``` being the baseline:

 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama27b_titanrtx.png" alt="Titan RTX" style="width:48%">
	<img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama270b_a100.png" alt="A100" style="width:48%">
  </div>
 </center>
</div> 

Additionally, we support external backends for faster inference with fused kernels. You can use these backends after the model was quantized as follows:
```Python
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="torchao_int4") #torchao's int4mm kernel, use compute_dtype=bfloat16
prepare_for_inference(model, backend="marlin", allow_merge=True) #marlin int4 kernel.
```
These backends only work with 4-bit quantization and `axis=1`. Additionally, for <a href="https://github.com/IST-DASLab/marlin.git">Marlin</a>, we only support `group_size=None`. Below you can find a comparison between the different backends. The torchao kernel reaches 184 tokens/sec on a 4090.

<p align="center">
    <img src="https://github.com/mobiusml/hqq/blob/master/imgs/llama_int4_4090.png" alt="backend 4090" >
</p>

### Supported Models
#### LLMs 
- Llama (Hugging Face + VLLM) 🦙
- Mistral (Hugging Face)
- Mixtral-8x7B (Hugging Face)
- Phi + Phi_opt (Hugging Face)

#### Vision 
- ViT-CLIP (timm) 🖼️

#### Auto Mode
- Hugging Face

### Hugging Face 🤗
First, make sure you have your Hugging Face token properly set via:
```
huggingface-cli login --token <your-token>
```
#### Basic Usage
You can quantize a Hugging Face model as follows:
```Python
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer

#Model and setttings
model_id      = 'meta-llama/Llama-2-7b-chat-hf'
compute_dtype = torch.float16
device        = 'cuda:0'

#Load model on the CPU
######################
model     = HQQModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
######################
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
model.quantize_model(quant_config=quant_config, compute_dtype=compute_dtype, device=device) 

```

You can save/load a quantized model as follows:
```Python
#Save the quantized model
model.save_quantized(save_dir=save_dir)

#Load from local directory or Hugging Face Hub on a specific device
model = HQQModelForCausalLM.from_quantized(save_dir_or_hfhub, device='cuda')
```

#### Multimodal
For multimodal models, you can quantize the models separately. Here's an example that quantizes the Llama language model in Llava:
```Python
#Load the model on CPU
import transformers
model_id      = "llava-hf/llava-1.5-13b-hf"
compute_dtype = torch.float16
device        = 'cuda:0'

processor = transformers.AutoProcessor.from_pretrained(model_id)
model     = transformers.LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=compute_dtype)

#Quantize and offload to GPU
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
LlamaHQQ.quantize_model(model.language_model, quant_config=quant_config, 
                                              compute_dtype=compute_dtype, 
                                              device=device)

#Use fp16 CLIP and tower
model.vision_tower          = model.vision_tower.to(device=device, dtype=compute_dtype)
model.multi_modal_projector = model.multi_modal_projector.to(device=device, dtype=compute_dtype)
model                       = model.eval();

#Optimize/compile (Optional)
model.vision_tower          = torch.compile(model.vision_tower)
model.multi_modal_projector = torch.compile(model.multi_modal_projector)
```

#### Auto Mode
If the model architecture is not manally defined in ```hqq/models/hf```, you can try the automatic mode that doesn't require knowing the architecture in advance:
```Python
from hqq.models.hf.base import AutoHQQHFModel

#Quantize
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, 
                                    compute_dtype=compute_dtype, 
                                    device=device)

#Save
AutoHQQHFModel.save_quantized(model, save_dir)

#Load
model = AutoHQQHFModel.from_quantized(save_dir)
```

### VLLM (Experimental)
By default, VLLM is not installed to avoid CUDA version problems. Make sure you install the right version that matches your CUDA settings (vllm <= 0.2.2): 
https://docs.vllm.ai/en/latest/getting_started/installation.html 

#### Basic Usage
After installation, you can quantize VLLM models as follows:

```Python
from hqq.engine.vllm import HQQLLM
model_id = 'meta-llama/Llama-2-7b-chat-hf'

#Loads the model (on CPU)
######################
model = HQQLLM(model=model_id)

#Quantize the model and dispatch on GPU
######################
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
model.quantize_model(quant_config=quant_config)
```

#### Langchain
Additionally, you can use the quantized model in Langchain (requires ```pip install langchain```) as follows:

```Python
from hqq.engine.vllm import LangchainVLLM
llm = LangchainVLLM(max_new_tokens=1000, top_p=0.90, temperature=0.6).set(model)
print(llm("Who is Elon Musk?"))
```

You can save/load a quantized model as follows:
```Python
#Save the quantized model
model.save_quantized(save_dir=save_dir)

#Load from local directory or Hugging Face Hub
model = HQQLLM.from_quantized(save_dir_or_hfhub)
```

Notes:
- Support is broken since post 0.2.2 update.
- The VLLM backend only works with a single GPU for now.
- Only VLLM models created via ```save_quantized``` can be loaded with ```HQQLLM.from_quantized```.

### Timm 🖼️
Timm backend is also supported. Here's how you use it:

```Python
model_id = 'vit_large_patch14_clip_224.laion2b'

#Load model on the CPU
######################
from hqq.engine.timm import HQQtimm
model = HQQtimm.create_model(model_id, pretrained=True)

#Quantize the model
######################
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
model.quantize_model(quant_config=quant_config, compute_dtype=torch.float16)

```

You can save/load the quantized models as follows:
```Python
#Save the quantized model
model.save_quantized(save_dir=save_dir)

#Load from local directory or Hugging Face Hub
model = HQQtimm.from_quantized(save_dir_or_hfhub)
```

### Quantize Custom Models 🗜️
If you want to quantize your own model architecture, you need to write a patching logic that goes through all the linear layers and replaces them with ```HQQLinear```. You can follow the examples provided in ```hqq/models```.

### Custom Quantization Configurations ⚙️
You can specify different quantization configs for different layers by feeding a dictionary in the form ```linear_tag: BaseQuantizeConfig()```, The following example uses 4-bit for ```self_attn.v_proj``` and 2-bit for the rest of the layers:
```Python
from hqq.core.quantize import *
q2_config    = BaseQuantizeConfig(nbits=2, group_size=16) #2-bit config
q4_config    = BaseQuantizeConfig(nbits=4, group_size=64) #4-bit config

linear_tags  = HQQModelForCausalLM.get_linear_tags(model) #List of tags for the linear layers of the model
quant_config = {k: q2_config for k in linear_tags}
quant_config['self_attn.v_proj'] = q4_config
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

#Optional: faster but might not work on older GPUs
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)

#Train ....

#Convert LoRA weights to the same model dtype for faster inference
model.eval()
PeftUtils.cast_lora_weights(model, dtype=torch.float16)

#Save LoRA weights
PeftUtils.save_lora_weights(model, filename)

#Load LoRA weights: automatically calls add_lora 
PeftUtils.load_lora_weights(model, filename)
```

We provide a complete example to train a model with HQQ/LoRA that you can find in ```examples/lora/train_hqq_lora_example.py```.

If you want to use muti-gpu training via FSDP, check out this awesome repo by Answer.AI: https://github.com/AnswerDotAI/fsdp_qlora

### Examples 
We provide a variety of examples demonstrating model quantization across different backends within the ```examples```  directory.

In the ```examples/llama2_benchmark```directory, you'll find code to replicate our Llama2 benchmark. By default, this benchmark quantizes the Llama2-7B model with 4-bit precision and provides perplexity metrics on wikitext-2.

To execute the benchmark, ensure you have the datasets package installed by running  ```pip install datasets```. Additionally, for the GPTQ and AWQ demos, you'll need to install the following packages: ```pip install auto-gptq[triton]==0.4.2 autoawq==0.1.4 triton==2.0.0```

After installation, configure your Hugging Face 🤗 token either through the command line or within the demo files, and you're all set!

### Citation 📜
```
@misc{badri2023hqq,
title  = {Half-Quadratic Quantization of Large Machine Learning Models},
url    = {https://mobiusml.github.io/hqq_blog/},
author = {Hicham Badri and Appu Shaji},
month  = {November},
year   = {2023}
```


