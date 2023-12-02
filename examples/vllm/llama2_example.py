model_id = 'meta-llama/Llama-2-7b-chat-hf'

#Load VLLM un-quantized model 
from hqq.engine.vllm import HQQLLM
model = HQQLLM(model=model_id)

#Quantize the model
from hqq.core.quantize import *
model.quantize_model(BaseQuantizeConfig(nbits=4, group_size=64))

#Optional: Save the model
#model.save_quantized(model_id.split('/')[-1] + '_quantized')

#Optional: Set backend
HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE) #set backend

#Generation
from vllm.entrypoints.llm import SamplingParams
sampling_params = SamplingParams(temperature=0.6, top_p=0.90, max_tokens=1000, repetition_penalty=1.2)

prompt = "How can I build a car?"

output = model.generate([prompt], sampling_params)[0]
print(output.prompt)
print(output.outputs[0].text)