#Settings
#pip install hqq==1.8.0
#pip install trl==
#pip install transformers==4.40.0
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ipython3 

#HQQ+ version with SFT. Note that we actually use model distillation for HQQ+ not SFT.

######################################################################################
import torch
cache_path    = '' 
model_id      = "meta-llama/Llama-2-7b-hf" 
compute_dtype = torch.bfloat16
device        = 'cuda:0'

#HQQ Quantize
######################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

model     = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path) 

#Quantize the model
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=2, group_size=8, quant_scale=False, quant_zero=False, axis=0)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Add Peft
######################################################################################
from hqq.core.peft import PeftUtils

train_dtype       = torch.torch.float32
atten_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':32, 'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}
mlp_lora_params   = {'lora_type':'default', 'r':8,  'lora_alpha':8,  'dropout':0.05, 'train_dtype':train_dtype, 'train_bias':True}

lora_params       = {'self_attn.q_proj': atten_lora_params,
                     'self_attn.k_proj': atten_lora_params,
                     'self_attn.v_proj': atten_lora_params,
                     'self_attn.o_proj': atten_lora_params,
                     'mlp.gate_proj'   : mlp_lora_params,
                     'mlp.up_proj'     : mlp_lora_params,
                     'mlp.down_proj'   : mlp_lora_params}
#Apply LoRA
PeftUtils.add_lora(model, lora_params)
HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
model.config.use_cache = False

#Dataset 
######################################################################################
from datasets import load_dataset, Dataset
from tqdm import tqdm
import transformers
import numpy as np 
import random

tokenizer.pad_token     = tokenizer.unk_token #tokenizer.eos_token 
tokenizer.padding_side  = "right" 
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#####################################################################################
#Train
from trl import SFTTrainer

#Play with these parameters 
grad_acc   = 1
logging_st = 1
max_steps  = -1
lr         = 1e-5 
batch_size = 1
n_epochs   = 2
max_tokens = 1024 

training_args = transformers.TrainingArguments(
    output_dir='.',	
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    learning_rate=lr,
    logging_steps=logging_st,
    num_train_epochs=n_epochs,
    max_steps=max_steps,
    remove_unused_columns=False,
    bf16=True,
    max_grad_norm=1.0,
    save_steps=10000000,
    lr_scheduler_type= "cosine", 
)

#Wrap model to avoid accelerate issues 
class WrappedModel(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, *args, **kwargs):
		return self.model.forward(*args, **kwargs)

	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()

	def parameters(self):
		return self.model.parameters()

trainer = SFTTrainer(
    model=WrappedModel(model),
    tokenizer=tokenizer,
    max_seq_length=max_tokens,
    train_dataset=dataset,
    eval_dataset=None,
    peft_config=None,
    args=training_args,
    dataset_text_field="text",
    packing=True,
)

model.is_parallelizable       = False
trainer.is_model_parallel     = False
trainer.place_model_on_device = False
model.train()
trainer.train()

# #Prediction/Eval
# ######################################################################################
from datasets import load_dataset
import torch, time
import numpy as np
from tqdm import tqdm
import gc

tokenizer.add_bos_token = True
tokenizer.add_eos_token = False
PeftUtils.cast_lora_weights(model, dtype=compute_dtype)
model.eval()

#Save lora weights
#PeftUtils.save_lora_weights(model, filename)

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

#Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html
def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True):
    model.eval()
    tokenizer.pad_token     = tokenizer.eos_token 
    tokenizer.padding_side  = "right" 
    tokenizer.add_eos_token = False

    dataset   = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')
    
    encodings['input_ids'] = encodings['input_ids'].to('cuda')

    lls, t = [], []
    for i in tqdm(range(0, encodings['input_ids'].size(1), stride), disable=not verbose):
        begin_loc  = max(i + stride - max_length, 0)
        end_loc    = min(i + stride, encodings['input_ids'].size(1))
        trg_len    = end_loc - i  
        input_ids  = encodings['input_ids'][:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100 #ignore context 

        t1 = time.time()
        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
        torch.cuda.synchronize()
        t2 = time.time()
        t.append((t2-t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl       = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if(verbose):
        print('perplexity', ppl)
        print('time', str(pred_time) + '  sec')

    del encodings
    cleanup()

    return {'perplexity':ppl, 'prediction_time':pred_time}


print('perplexity',eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True))
