# Tested with torch nightly, 4090 
# pip uninstall torch -y; pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
##############################################################################################

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id      = "openai/whisper-medium"
#model_id     = "distil-whisper/distil-large-v3"

compute_dtype = torch.bfloat16
device        = "cuda:0"

model     = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=compute_dtype) 
processor = AutoProcessor.from_pretrained(model_id)


##############################################################################
#No quantize
#model = model.to(device)

##############################################################################
#Quantize
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1) 
HQQLinear.set_backend(HQQBackend.PYTORCH)

AutoHQQHFModel.quantize_model(model.model.encoder, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
AutoHQQHFModel.quantize_model(model.model.decoder, quant_config=quant_config, compute_dtype=compute_dtype, device=device)

from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model.model.encoder)
prepare_for_inference(model.model.decoder, backend="torchao_int4")

model.model.encoder.forward = torch.compile(model.model.encoder.forward, mode="reduce-overhead", fullgraph=True)
model.model.decoder.forward = torch.compile(model.model.decoder.forward, mode="reduce-overhead", fullgraph=True)
# ##############################################################################

import time 
import numpy as np 

if(model_id=="openai/whisper-medium"):
	encoder_input = torch.randn([1, 80, 3000], dtype=compute_dtype, device=device)
if(model_id=="distil-whisper/distil-large-v3"):
	encoder_input = torch.randn([1, 128, 3000], dtype=compute_dtype, device=device)

def run_encoder():
	with torch.no_grad():
		model.model.encoder(encoder_input)
	torch.cuda.synchronize()

t = []
for _ in range(200):
	t1 = time.time()
	run_encoder()
	t2 = time.time()
	t.append(t2-t1)
print("Encoder", np.mean(t[-100:]), "sec / sample")


decoder_input = torch.randint(0, 1000, [1, 1], dtype=torch.int64, device=device)
def run_decoder():
	with torch.no_grad():
		out = model.model.decoder(decoder_input)
	torch.cuda.synchronize()


t = []
for _ in range(200):
	t1 = time.time()
	run_decoder()
	t2 = time.time()
	t.append(t2-t1)
print("Decoder", np.mean(t[-100:]), "sec / sample")


#openai/whisper-medium | RTX 4090
#Encoder: use default backend
#fp16                         : 0.0234 sec / sample
#hqq 4-bit (default,compiled) : 0.0124 sec / sample | 1.89x faster 


#Decoder: use torchao backend to decode 1 token at a time
#fp16                         : 0.01080 sec / sample
#hqq 4-bit (ao_int4, compiled): 0.000928 sec / sample |  11.63x faster


#distil-whisper/distil-large-v3 | RTX 4090
#Encoder: use default backend
#fp16                         : 0.03738  sec / sample
#hqq 4-bit (default,compiled) : 0.01869 sec / sample | 2x faster 


#Decoder: use torchao backend to decode 1 token at a time
#fp16                         : 0.002592 sec / sample
#hqq 4-bit (ao_int4, compiled): 0.000326 sec / sample |  7.95x faster
