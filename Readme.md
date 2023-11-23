## Half-Quadratic Quantization (HQQ)
This folder contains the code to perform Half-Quadratic Quantization (<b>HQQ</b>) quantization and reproduce the results from our blogpost: https://mobiusml.github.io/hqq_blog/ 

### Installation 
You can install the core library directly via ```pip install git+https://github.com/mobiusml/hqq.git```. 

Alternatively, clone the repo and run ```pip install .``` from this current folder. 

### Important ⚠️
If you are using a virtual machine on the cloud, make sure you limit the number of threads to only those available. Otherwise, processing will be unusually slow, especially for the GPTQ benchmark. You can do that by limiting the OMP threads:
```
num_threads=32; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 python 
```

### Basic Usage
To perform quantization via HQQ, you simply need to replace the ```torch.nn.Linear``` layers as follows:
```Python
from hqq.core.quantize import *
#Quantization settings
quant_config = hqq_base_quant_config(nbits=4, group_size=64)
#Replace linear layer
hqq_layer = HQQLinear(your_linear_layer, quant_config, del_orig=True)
#del_orig=True will remove the original linear layer from memory
```

### LLama2 Quantization 🦙
First, make sure to install the following dependencies:
```pip install transformers[torch] datasets xformers accelerate```

You can quantize a LLama2 HuggingFace model as follows:

```Python
import torch, transformers
model_id  = "meta-llama/Llama-2-7b-hf" 

#Load model on the CPU
######################
model     = transformers.AutoModelForCausalLM.from_pretrained(model_id) 
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
######################
from hqq.core.quantize import hqq_base_quant_config
from hqq.models.llama  import LlamaHQQ

quant_config = hqq_base_quant_config(nbits=4, group_size=64)
LlamaHQQ.quantize_model(model, quant_config=quant_config)
```

You can save/load the quantized models as follows:
```Python
#Save
LlamaHQQ.save_quantized(model, save_dir=save_dir)
#Load from local directory or Hugging Face Hub
model = LlamaHQQ.from_quantized(save_dir)
```
We provide a complete example to quantize LLama2 models that you can find in the ```llama2_benchmark``` folder. By default, it quantizes the LLama2-7B model with 4-bit precision and reports the perplexity on wikitext-2. 

Additionally, to run the GPTQ and AWQ demos you need the following:
```pip install auto-gptq[triton]==0.4.2 autoawq==0.1.4 triton==2.0.0```

Then set your HuggingFace 🤗 token via cli or inside the demo files, and you're all set!

### ViT Quantization 🖼️
Make sure to install _timm_ via ```pip install timm``` first. 

You can quantize a ViT model as follows:
```Python
import timm, torch
model_id = 'vit_large_patch14_clip_224.laion2b'

#Load model on CPU
model = timm.create_model(model_id, pretrained=True)

#Quantize
from hqq.core.quantize import hqq_base_quant_config
from hqq.models.vit import ViTHQQ
quant_config = hqq_base_quant_config(nbits=4, group_size=64)
ViTHQQ.quantize_model(model, quant_config=quant_config)
```

You can also save/load the quantized ViT models as follows:
```Python
#Save
ViTHQQ.save_quantized(model, save_dir=save_dir)
#Load from local directory or Hugging Face Hub
model = ViTHQQ.from_quantized(save_dir)
```

We provide a complete example to quantize ViT models that you can find in the ```vit_example``` folder. The script shows how to quantize a _timm_ ViT model and compares the dot score between the quantized and the original model predictions.

  
### Quantize Custom Models 🗜️
If you want to quantize your own model architecture, you need to write a patching function that goes through all the linear layers and replaces them with ```HQQLinear```. You can follow the examples provided in ```hqq/models```.

### Models from HuggingFace Hub 🤗
We provide pre-quantized LLama2 models that you can directly use from [HuggingFace Hub](https://huggingface.co/mobiuslabsgmbh): 

Here's an example:
```Python
import transformers
from hqq.models.llama import LlamaHQQ

model_id = 'mobiuslabsgmbh/Llama-2-7b-hf-4bit_g64-HQQ'
#Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
#Load the model
model = LlamaHQQ.from_quantized(model_id)
```



