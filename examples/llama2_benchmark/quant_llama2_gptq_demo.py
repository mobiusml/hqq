import torch, transformers

#Important: limit the number of threads otherwise the process will hang for a long time
#num_threads=32; OMP_NUM_THREADS=$num_threads OPENBLAS_NUM_THREADS=$num_threads MKL_NUM_THREADS=$num_threads VECLIB_MAXIMUM_THREADS=$num_threads NUMEXPR_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = "meta-llama/Llama-2-7b-hf" 
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

#GPTQ settings
######################################################################################
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging, gc, time
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

#Adapted from: https://towardsdatascience.com/4-bit-quantization-with-gptq-36b0f4f02c34
def prepare_model(model, tokenizer, n_samples=1024, max_tokens=512, use_triton=True):
	# Load data and tokenize examples
	from datasets import load_dataset
	import random
	data           = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples}]", cache_dir=cache_path)
	tokenized_data = torch.cat([tokenizer(data[i]['text'], return_tensors='pt').input_ids for i in tqdm(range(len(data)))], axis=-1) #~536K tokens

	# Format tokenized examples
	random.seed(1) 
	examples_ids = []
	for _ in range(n_samples):
		i              = random.randint(0, tokenized_data.shape[1] - max_tokens - 1)
		j              = i + max_tokens
		input_ids      = tokenized_data[:, i:j]
		attention_mask = torch.ones_like(input_ids)
		examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

	print('Using ' + str(len(examples_ids)) + ' samples for calibration.')
	model.quantize(examples_ids, batch_size=1, use_triton=use_triton)
	model = model.cuda(); 
	with torch.no_grad(): x = model(input_ids.to('cuda'));
	del examples_ids, x
	torch.cuda.empty_cache()
	gc.collect()
	return model

#quantize_config = BaseQuantizeConfig(bits=8, group_size=128, damp_percent=0.01, desc_act=False); use_triton=True;
#quantize_config = BaseQuantizeConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=False); use_triton=True;
quantize_config = BaseQuantizeConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False); use_triton=True;
#quantize_config = BaseQuantizeConfig(bits=3, group_size=128, damp_percent=0.01, desc_act=False); use_triton=False;
#quantize_config = BaseQuantizeConfig(bits=3, group_size=64, damp_percent=0.01, desc_act=False); use_triton=False;
#quantize_config = BaseQuantizeConfig(bits=2, group_size=64, damp_percent=0.01, desc_act=False); use_triton=True;

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,           use_auth_token=hf_auth)
model     = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config, use_auth_token=hf_auth, cache_dir=cache_path)
t1 = time.time()
model = prepare_model(model, tokenizer, use_triton=use_triton)
t2 = time.time()
print('Took ' + str(t2-t1) + ' seconds to quantize the model with GPTQ')

#Evaluate the quantized model 
######################################################################################
from eval_model import eval_wikitext2

eval_wikitext2(model, tokenizer, verbose=True)

