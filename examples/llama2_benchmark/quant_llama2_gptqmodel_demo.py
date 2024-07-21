# Install GPTQModel from source via https://github.com/ModelCloud/GPTQModel#Install
# This is different from gptq_demo.py as it uses GPTQModel (instead of AutoGPTQ) to more efficiently quantize to GPTQ format while
# also at the same creating a higher quality quant.
# Diff vs AutoGPTQ code:
# 1. batching
# 2. Pass dataset as simple str: no need for complicated and often incorrect tokenization/attention mask code
# 3. desc_act set to True
# 4. damp set to 0.005 instead of 0.01
# Result of this code on A100 80GB: PPL = 5.3972, Quantization Time (10m quant + 3m pack):  721 seconds

import transformers
import logging
import time

from eval_model import eval_wikitext2
from gptqmodel import GPTQModel, QuantizeConfig

# Settings
######################################################################################
hf_auth = None  # HuggingFace token
cache_path = ''  # cache directory to store data


# Chose a model
model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "meta-llama/Llama-2-13b-hf"
# model_id = "meta-llama/Llama-2-70b-hf"

# GPTQModel settings
######################################################################################
logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def prepare_model(model, tokenizer, n_samples=1024, max_length=512):
	# Load data and tokenize examples
	from datasets import load_dataset

	calib_data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train").filter(lambda x: len(x["text"]) >= max_length//2 and len(x["text"]) <= max_length)
	calib_data = [tokenizer(row["text"]) for row in calib_data.select(range(n_samples))]

	print('Using ' + str(len(calib_data)) + ' samples for calibration.')
	model.quantize(calib_data, batch_size=64)
	return model

# quantize_config = QuantizeConfig(bits=8, group_size=128, damp_percent=0.01, desc_act=False)
quantize_config = QuantizeConfig(bits=4, group_size=128, damp_percent=0.005, desc_act=True)
# quantize_config = QuantizeConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False)
# quantize_config = QuantizeConfig(bits=3, group_size=128, damp_percent=0.01, desc_act=False)
# quantize_config = QuantizeConfig(bits=3, group_size=64, damp_percent=0.01, desc_act=False)
# quantize_config = QuantizeConfig(bits=2, group_size=64, damp_percent=0.01, desc_act=False)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = GPTQModel.from_pretrained(model_id, quantize_config)
t1 = time.time()
model = prepare_model(model, tokenizer)
t2 = time.time()
print('Took ' + str(t2-t1) + ' seconds to quantize the model with GPTQ')

# Evaluate the quantized model
######################################################################################
model = model.to(device="cuda")
eval_wikitext2(model, tokenizer, verbose=True)
