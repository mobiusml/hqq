## Half-Quadratic Quantization (HQQ)
This folder contains the code to perform Half-Quadratic Quantization (<b>HQQ</b>) presented in our article: https://mobiusml.github.io/hqq_blog/ 

### WHat is HQQ?
<b>HQQ</b> is a fast and accurate model quantizer that skips the need for calibration data. It's super simple to implement (just a few lines of code for the optimizer). It can crunch through quantizing the Llama2-70B model in only 4 minutes! 🚀

### Installation 
First, make sure you have a Pytorch 2 version that matches your CUDA version: https://pytorch.org/

You can install hqq via  ```pip install hqq```. 

To get the latest version, you can install the core library directly via ```pip install git+https://github.com/mobiusml/hqq.git```. 

Alternatively, clone the repo and run ```pip install .``` from this current folder. 

### Important ⚠️
If you are using a virtual machine on the cloud, make sure you limit the number of threads to only those available. Otherwise, processing will be unusually slow, especially for the GPTQ benchmark. You can do that by limiting the OMP threads:
```
num_threads=32; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 python 
```

### Basic Usage
To perform quantization with HQQ, you simply need to replace the linear layers ( ```torch.nn.Linear```) as follows:
```Python
from hqq.core.quantize import *
#Quantization settings
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False)
#Replace your linear layer 
hqq_layer = HQQLinear(your_linear_layer, quant_config, del_orig=True)
#del_orig=True will remove the original linear layer from memory
```

The quantization parameters are set as follows:

- ```nbits``` (int): supports 8, 4, 3, 2, 1 bits.
- ```group_size``` (int): no restrictions as long as ```weight.numel()``` is divisible by the ```group_size```.
- ```quant_zero``` (bool): if True, it quantizes the zero-point to 8-bit without grouping.
- ```quant_scale``` (bool): if True, it quantizes the scaling factor to 8-bit with a group_size of 128.

Additionally, you can set ```offload_meta=True``` to offload the meta-data to the CPU. This drastically decreases the GPU memory requirements but makes processing slightly slower for smaller group-sizes. With ```offload_meta=True```, you can run Llama2-70B and Mixtral with HQQ 2-bit using only 18.8GB and 13GB VRAM respectively!

You can try to change the backend which could speed-up the runtime:
```Python
HQQLinear.set_backend(HQQBackend.PYTORCH)                  #Pytorch backend (default) 
HQQLinear.set_backend(HQQBackend.PYTORCH_BACKPROP)         #Same as BACKPROP but supports the backward pass
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)          #Compiled Pytorch
HQQLinear.set_backend(HQQBackend.PYTORCH_BACKPROP_COMPILE) #Same as PYTORCH_COMPILE, but supports the backward pass
HQQLinear.set_backend(HQQBackend.ATEN)                     #C++ Aten/Torch backend (CUDA and Pytorch)
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)            #Same as ATEN but supports the backward pass 
```
We recommend you use the ```HQQBackend.ATEN_BACKPROP``` backend for faster processing. You can install as follows:
```
cd hqq/kernels && python setup_cuda.py install;
```

The ```HQQBackend.ATEN_BACKPROP``` backend with ```setup_cuda``` uses CUDA kernels for the dequantization step. This leads to a significant speed-up compared to ```PYTORCH_COMPILE``` and can be combined with ```torch.compile(model)``` for even faster runtime:

 <div class="row"><center>
  <div class="column">
    <img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama27b_titanrtx.png" alt="Titan RTX" style="width:48%">
	<img src="https://github.com/mobiusml/hqq/blob/master/imgs/hqq_cuda_dequant_llama270b_a100.png" alt="A100" style="width:48%">
  </div>
 </center>
</div> 

### Supported Models
#### LLMs 
- Llama (Hugging Face + VLLM) 🦙
- Mistral (Hugging Face)
- Mixtral-8x7B (Hugging Face)
- Phi + Phi_opt (Hugging Face)

#### Vision 
- ViT-CLIP (timm) 🖼️

### Hugging Face 🤗
First, make sure you have your Hugging Face token properly set via:
```
huggingface-cli login --token <your-token>
```

You can quantize a Hugging Face model as follows:
```Python
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
model_id   = 'meta-llama/Llama-2-7b-chat-hf'

#Load model on the CPU
######################
model     = HQQModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
######################
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
model.quantize_model(quant_config=quant_config, compute_dtype=torch.float16) 

#Optional: set backend
######################
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
```

You can save/load a quantized model as follows:
```Python
#Save the quantized model
model.save_quantized(model, save_dir=save_dir)
#Load from local directory or Hugging Face Hub
model = HQQModelForCausalLM.from_quantized(save_dir_or_hfhub)
```

Alternatively, you can also work with models created via ```transformers.AutoModelForCausalLM```:
```Python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id) 
#Quantize
HQQModelForCausalLM.quantize_model_(model, quant_config=quant_config)
```

For multimodal models, you can quantize the models separately. Here's an example that quantizes the Llama language model in Llava:
```Python
#Load the model on CPU
import transformers
model_id  = "llava-hf/llava-1.5-13b-hf",
processor = transformers.AutoProcessor.from_pretrained(model_id)
model     = transformers.LlavaForConditionalGeneration.from_pretrained(model_id)

#Quantize and offload to GPU
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ
LlamaHQQ.quantize_model(model.language_model, quant_config=BaseQuantizeConfig(nbits=4, group_size=64))

#Use fp16 CLIP and tower
model.vision_tower          = model.vision_tower.half().cuda()
model.multi_modal_projector = model.multi_modal_projector.half().cuda()
model                       = model.eval();

#Optimize/compile (Optional)
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
model.vision_tower          = torch.compile(model.vision_tower)
model.multi_modal_projector = torch.compile(model.multi_modal_projector)
```

### VLLM 🚀
By default, VLLM is not installed to avoid CUDA version problems. Make sure you install the right version that matches your CUDA settings (vllm <= 0.2.2): 
https://docs.vllm.ai/en/latest/getting_started/installation.html 

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

#Optional: set backend
######################
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
```

Additionally, you can use the quantized model in Langchain (requires ```pip install langchain```) as follows:

```Python
from hqq.engine.vllm import LangchainVLLM
llm = LangchainVLLM(max_new_tokens=1000, top_p=0.90, temperature=0.6).set(model)
print(llm("Who is Elon Musk?"))
```

You can save/load a quantized model as follows:
```Python
#Save the quantized model
model.save_quantized(model, save_dir=save_dir)
#Load from local directory or Hugging Face Hub
model = HQQLLM.from_quantized(save_dir_or_hfhub)
```

Notes:
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

#Optional: set backend
######################
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
```

You can save/load the quantized models as follows:
```Python
#Save the quantized model
model.save_quantized(model, save_dir=save_dir)
#Load from local directory or Hugging Face Hub
model = HQQtimm.from_quantized(save_dir_or_hfhub)
```

### Quantize Custom Models 🗜️
If you want to quantize your own model architecture, you need to write a patching logic that goes through all the linear layers and replaces them with ```HQQLinear```. You can follow the examples provided in ```hqq/models```.

### Custom Quantization Configurations ⚙️
You can specify different quantization configs for different layers by feeding a dictionary in the form ```linear_tag: BaseQuantizeConfig()```, The following example uses 4-bit for ```self_attn.v_proj``` and 2-bit for the rest of the layers:
```Python
from hqq.core.quantize import *
linear_tags  = HQQModelForCausalLM.get_linear_tags(model) #List of tags for the linear layers of the model
q2_config    = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True) #2-bit config
q4_config    = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False) #4-bit config
quant_config = dict([(k, q2_config) for k in linear_tags])
quant_config['self_attn.v_proj'] = q4_config

#Quantize 
model.quantize_model(quant_config=quant_config)
```

### LoRA Training
You can use HQQ for lora training as follows:
```Python
#First, quantize/load a quantized HQQ model the
from hqq.core.peft import PeftUtils

base_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':64, 'dropout':0.05, 'train_dtype':torch.bfloat16}
lora_params      = {'self_attn.q_proj': base_lora_params,
                    'self_attn.k_proj': base_lora_params,
                    'self_attn.v_proj': base_lora_params,
                    'self_attn.o_proj': base_lora_params,
                    'mlp.gate_proj'   : None,
                    'mlp.up_proj'     : None,
                    'mlp.down_proj'   : None}


PeftUtils.add_lora(model, lora_params)

#Optional: faster but might not work on older GPUs
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)

#Train ....

#Convert lora weights to the same model dtype for faster inference
model.eval()
PeftUtils.cast_lora_weights(model, dtype=torch.half)
```

We provide a complete example to train a model with HQQ/LoRA that you can find in ```examples/lora/train_hqq_lora_example.py```.

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


