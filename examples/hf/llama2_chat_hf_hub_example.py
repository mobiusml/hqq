model_id  = 'mobiuslabsgmbh/Llama-2-7b-chat-hf-4bit_g64-HQQ'
#model_id = 'mobiuslabsgmbh/Llama-2-13b-chat-hf-4bit_g64-HQQ'
#model_id = 'mobiuslabsgmbh/Llama-2-70b-chat-hf-2bit_g16_s128-HQQ'

from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = HQQModelForCausalLM.from_quantized(model_id)

##########################################################################################################
import transformers
from threading import Thread

from sys import stdout
def print_flush(data):
    stdout.write("\r" + data)
    stdout.flush()

#Adapted from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py
def process_conversation(chat):
    system_prompt = chat['system_prompt']
    chat_history  = chat['chat_history']
    message       = chat['message']

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    return tokenizer.apply_chat_template(conversation, return_tensors="pt").to('cuda')

def chat_processor(chat, max_new_tokens=100, do_sample=True):
    tokenizer.use_default_system_prompt = False
    streamer = transformers.TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_params = dict(
        {"input_ids": process_conversation(chat)},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=0.90,
        top_k=50,
        temperature= 0.6,
        num_beams=1,
        repetition_penalty=1.2,
    )

    t = Thread(target=model.generate, kwargs=generate_params)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        print_flush("".join(outputs))

    return outputs

###################################################################################################

outputs = chat_processor({'system_prompt':"You are a helpful assistant.",
                        'chat_history':[],
                        'message':"How can I build a car?"
                        }, 
                         max_new_tokens=1000, do_sample=False)
