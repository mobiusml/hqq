#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = "meta-llama/Llama-2-7b-hf" 
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

#HQQ Quantize
######################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
model     = HQQModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, cache_dir=cache_path)
tokenizer = AutoTokenizer.from_pretrained(model_id,       use_auth_token=hf_auth, cache_dir=cache_path) 

#Quantize the model
from hqq.core.quantize import *
quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False)
model.quantize_model(quant_config=quant_config)

#Add Peft
######################################################################################
from hqq.core.peft import PeftUtils
from hqq.core.quantize import *

train_dtype      = torch.float32 
base_lora_params = {'lora_type':'default', 'r':32, 'lora_alpha':64, 'dropout':0.05, 'train_dtype':train_dtype}
lora_params      = {'self_attn.q_proj': base_lora_params,
		    'self_attn.k_proj': base_lora_params,
		    'self_attn.v_proj': base_lora_params,
		    'self_attn.o_proj': base_lora_params,
		    'mlp.gate_proj'   : None,
		    'mlp.up_proj'     : None,
		    'mlp.down_proj'   : None}

#Apply LoRA
PeftUtils.add_lora(model, lora_params)

#Dataset 
######################################################################################
from datasets import load_dataset, Dataset
from tqdm import tqdm
import transformers
import numpy as np 
import random

tokenizer.pad_token     = tokenizer.eos_token 
tokenizer.padding_side  = "right" 
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

batch_size  = 1 
num_epochs  = 1
grad_acc    = 1
max_tokens  = 256 
max_samples = 5000

#OpenAssistant
##########################################################################
dataset     = load_dataset("timdettmers/openassistant-guanaco", split="train") 
dataset_val = load_dataset("timdettmers/openassistant-guanaco", split="test")  

def pre_process_chat(chat):
    #add proper chat preprocessing (bos/eos tokens, etc.)
	return chat

def assitant_prompt(prompt):
	return '### Human:' + prompt + '\n### Assistant:'


random.seed(100)
idx         = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
dataset     = Dataset.from_dict({'text':[pre_process_chat(dataset[i]['text']) for i in tqdm(idx)]})
dataset_val = Dataset.from_dict({'text':[pre_process_chat(dataset_val[i]['text']) for i in range(len(dataset_val))]})

#####################################################################################
#Train
from trl import SFTTrainer

grad_acc   = 2
logging_st = 1
max_steps  = -1
lr         = 1e-4 
batch_size = 1
n_epochs   = 1

training_args = transformers.TrainingArguments(
    output_dir='.',	
    per_device_train_batch_size=batch_size,
    #per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    learning_rate=lr,
    logging_steps=logging_st,
    num_train_epochs=n_epochs,
    max_steps=max_steps,
    #evaluation_strategy = "epoch",
    remove_unused_columns=False,
    #logging_strategy="epoch",
    fp16=train_dtype==torch.float32,
    max_grad_norm=1.0,
    save_steps=10000000,
    lr_scheduler_type= "linear", 
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
)

model.is_parallelizable       = False
trainer.is_model_parallel     = False
trainer.place_model_on_device = False

model.train()
trainer.train()

#Prediction/Eval
######################################################################################
#from #https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
def compute_perplexity_batched(model, tokenizer, predictions, encodings=None, batch_size=1, add_start_token=True, device='cuda', max_length=None):
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (len(existing_special_tokens) > 0), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (tokenizer.bos_token is not None), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length


    if(encodings is None):
        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks    = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index     = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask     = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch     = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask         = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1)

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1))

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)


tokenizer.add_bos_token = True
tokenizer.add_eos_token = False
model.eval()

#Convert lora weights to the same model dtype for faster inference
PeftUtils.cast_lora_weights(model, dtype=torch.half)

print('perplexity', compute_perplexity_batched(model=model, tokenizer=tokenizer, predictions=[s['text'] for s in dataset_val], batch_size=1, max_length=max_tokens))
