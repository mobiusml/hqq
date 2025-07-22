# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

# Rewrite generate() to support torch.compile fullgraph.
# https://gist.github.com/ArthurZucker/5dc54a3fb443e979fac437e5df7c800b

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Union, Dict
from transformers import StaticCache
from tqdm import tqdm

WARMUP_PROMPTS = [
    "Write an essay about large language models.",
    "Tell me a funny joke!",
    "How to make a yummy chocolate cake?",
    "Who is Elon Musk?",
    "Write a Python code snippet that adds two numbers together.",
]

#The original function from accelerate breaks with torch.compile, this is a hacky fix
def patch_accelerate_device_hook():
    #https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/operations.py#L136
    import accelerate.utils.operations as ops 
    from typing import Mapping
    def _send_to_device(tensor, device,  non_blocking=False, skip_keys=None):
        if(isinstance(tensor, tuple)):
            return tuple(t.to(device) for t in tensor)
        if(isinstance(tensor, list)):
            return [t.to(device) for t in tensor]
        if(isinstance(tensor, Mapping)):
            if isinstance(skip_keys, str):
                skip_keys = [skip_keys]
            elif skip_keys is None:
                skip_keys = []
            return type(tensor)(
                {
                    k: t if k in skip_keys else _send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                    for k, t in tensor.items()
                }
            )
        if(isinstance(tensor, (torch.Tensor, torch.nn.Parameter))):
            return tensor.to(device)
        else:
            return tensor

    ops.send_to_device = _send_to_device


# Patches a HuggingFace model to work with torch.compile + static cache
def patch_model_for_compiled_runtime(
    model, tokenizer, warmup=True, max_new_tokens=1000, patch_accelerate=True, pre_compile=None, compile_prefill=False,
):  
    if(patch_accelerate):
        patch_accelerate_device_hook()

    model.config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.eval()
    
    torch._dynamo.config.inline_inbuilt_nn_modules = False #torch 2.5.0 fix

    forward_simple = model.forward

    if(pre_compile is None):
        pre_compile = getattr(model.generation_config, 'compile_config', None) is None 

    if(pre_compile):
        if(compile_prefill):
            forward_prefill = torch.compile(model.forward, dynamic=True, fullgraph=True)
        else:
            forward_prefill = model.forward
        forward_decode = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    else:
        forward_prefill = model.forward
        forward_decode = model.forward

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    def custom_forward(*args, **kwargs):
        # Decoding pahse
        if (len(args) > 0 and args[0].shape[-1] == 1) or (
            "input_ids" in kwargs and kwargs["input_ids"].shape[-1] == 1
        ):
            return forward_decode(*args, **kwargs)
        else:
            #Prefill phase
            return forward_prefill(*args, **kwargs)

    model.forward = custom_forward

    # Warm-up
    if warmup:
        from tqdm import tqdm

        for prompt in tqdm(WARMUP_PROMPTS):
            model.generate(
                **tokenizer(
                    [
                        tokenizer.apply_chat_template(
                            [
                                {"role": "user", "content": prompt},
                            ],
                            tokenize=False,
                        )
                    ],
                    return_tensors="pt",
                ).to(model.device),
                max_new_tokens=max_new_tokens,
                cache_implementation="static",
                pad_token_id=tokenizer.pad_token_id,
            )


# Custom generator
class HFGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 1000,
        cache_size: Union[int, None] = None,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_k: int = 5,
        compile: Union[str, None] = None,  # None / "partial" / "full"
        compile_options: Dict = {"mode": "reduce-overhead", "fullgraph": True},
        patch_accelerate: bool = True,
    ):
        super().__init__()

        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.capture_scalar_outputs = True
        torch._inductor.config.fx_graph_cache = True
        try:
            torch._dynamo.config.inline_inbuilt_nn_modules = False #torch 2.5.0 fix
        except:
            pass

        if(patch_accelerate):
            patch_accelerate_device_hook()

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.do_sample = do_sample
        self.temperature = temperature if self.do_sample else None
        self.top_k = top_k if self.do_sample else None
        self.use_cache = True  # False
        self.compile_options = compile_options

        if do_sample:
            decode_one_token = self.decode_one_token_sampled
        else:
            decode_one_token = self.decode_one_token_no_sample

        # Setup cache
        self.max_new_tokens = max_new_tokens
        if cache_size is None:
            self.cache_size = self.next_multiple(self.max_new_tokens)
        else:
            self.cache_size = cache_size

        self.max_new_tokens = min(self.max_new_tokens, self.cache_size)

        self.setup_cache()

        self.is_compiled = False
        if compile == "partial":
            self.compile_partial(decode_one_token)

        if compile == "full":
            self.compile_full()

        if hasattr(self, "decode_one_token") is False:
            self.decode_one_token = decode_one_token

        self.init()  # check this: move this before setup_cache?

        ############################
        #Cuda Graph section
        self.static_input     = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.static_output    = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.cuda_graph       = None
        self.do_capture_graph = False
        ############################

    @torch.no_grad()
    def setup_cache(self):
        self.past_key_values = StaticCache(
            self.model.config, 1, self.cache_size, self.model.device, self.model.dtype
        )

    @torch.no_grad()
    def reset_cache(self):
        self.past_key_values.reset()

    # Ideally only compile this, but it creates issues with generation: https://github.com/huggingface/transformers/issues/30351
    def compile_partial(self, decode_one_token):
        self.decode_one_token = torch.compile(decode_one_token, **self.compile_options)
        self.is_compiled = True

    @torch.inference_mode()
    def compile_full(self):
        self.model.forward = torch.compile(self.model.forward, **self.compile_options)

        with sdpa_kernel([SDPBackend.MATH]):
            for ctx_size in [1] * 10:
                self.model(
                    torch.ones(
                        (1, ctx_size), dtype=torch.int32, device=self.model.device
                    ),
                    use_cache=False,
                ).logits  # check this: use_cache=True?
        self.is_compiled = True

    # torch.compile needs about 5-10 runs to warmup
    def warmup(self, max_samples=-1):
        for prompt in WARMUP_PROMPTS[:max_samples if max_samples>0 else len(WARMUP_PROMPTS)]:
            self.generate(prompt, print_tokens=False)
        return self

    def next_multiple(self, val):  # next power of 2
        vals = [2**i for i in range(5, 20)]  # [32, 64, ...]
        new_val = vals[[i for i in range(len(vals)) if (vals[i] - val) > 0][0]]
        return new_val

    def init(self):
        # Setup inference mode
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
        self.tokenizer.padding_side = "right"
        self.model.eval()
        self.model.generation_config.cache_implementation = "static"
        self.model.config.use_cache = True

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def multinomial_sample_one_no_sync(self, probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def logits_to_probs(self, logits, temperature=1.0, top_k=None):
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def sample(self, logits, temperature, top_k):
        probs = self.logits_to_probs(logits[:, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def decode_one_token_no_sample(
        self,
        cur_token,
        input_pos,
        cache_position,
        past_key_values,
        temperature=None,
        top_k=None,
    ):
        out = self.model(
            cur_token,
            # position_ids=input_pos,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        return new_token

    def decode_one_token_sampled(
        self,
        cur_token,
        input_pos,
        cache_position,
        past_key_values,
        temperature=0.6,
        top_k=5,
    ):
        out = self.model(
            cur_token,
            # position_ids=input_pos,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        new_token = self.sample(logits, temperature=temperature, top_k=top_k)[0]
        return new_token

    # Setup cache and variables
    def setup(self, inputs, max_new_tokens):
        self.reset_cache()
        self.inputs = inputs
        self.batch_size, self.seq_length = self.inputs["input_ids"].shape
        self.cache_position = torch.arange(self.seq_length, device=self.device)
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=self.device,
        )
        self.generated_ids[:, self.cache_position] = self.inputs["input_ids"].to(
            torch.int
        )

    # Pre-fill phase
    def prefill(self):
        out = self.model(
            **self.inputs,
            cache_position=self.cache_position,
            past_key_values=self.past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        self.generated_ids[:, self.seq_length] = next_token[:, 0]
        self.cache_position = torch.tensor([self.seq_length], device=self.device, dtype=torch.long)
        self.begin_gen_position = self.cache_position.item()
        return next_token

    # generate one token at a time
    def gen_next_token_raw(self, next_token):
        with sdpa_kernel([SDPBackend.MATH]):
            next_token = self.decode_one_token(
                next_token.clone(),
                None,
                cache_position=self.cache_position + 1,
                past_key_values=self.past_key_values,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def gen_next_token(self, next_token):
        return self.gen_next_token_raw(next_token)

    def enable_cuda_graph(self):
        #Warm-up
        self.warmup(1)

        #Enable 
        self.gen_next_token = self.gen_next_token_withgraph_v1
        self.do_capture_graph = True

        #Capture
        self.warmup(1)

        return self

    def gen_next_token_withgraph_v1(self, next_token):
        self.static_input.copy_(next_token)
        self.cache_position += 1

        if(self.do_capture_graph):
            self.stream = torch.cuda.Stream()
            torch.cuda.synchronize()

            #Warm-up
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    with sdpa_kernel([SDPBackend.MATH]):
                        _ = self.decode_one_token(
                            self.static_input,
                            None,
                            cache_position=self.cache_position,
                            past_key_values=self.past_key_values,
                            temperature=self.temperature,
                            top_k=self.top_k,
                        )
                torch.cuda.synchronize()
  
            #Capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                self.cuda_graph.capture_begin()
                with sdpa_kernel([SDPBackend.MATH]):
                    out = self.decode_one_token(
                        self.static_input,
                        None,
                        cache_position=self.cache_position,
                        past_key_values=self.past_key_values,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                    self.static_output.copy_(out)
                self.cuda_graph.capture_end()
            torch.cuda.synchronize()
            
            #Turn-off
            self.do_capture_graph = False
        else:
            self.cuda_graph.replay()

        next_token = self.static_output
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def gen_next_token_withgraph_v2(self, next_token):
        if(self.do_capture_graph):
            self.static_input.copy_(next_token)
            self.stream = torch.cuda.Stream()
            torch.cuda.synchronize()

            #Warm-up
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    with sdpa_kernel([SDPBackend.MATH]):
                        _ = self.decode_one_token(
                            self.static_input,
                            None,
                            cache_position=self.cache_position + 1,
                            past_key_values=self.past_key_values,
                            temperature=self.temperature,
                            top_k=self.top_k,
                        )
                torch.cuda.synchronize()
  
            #Capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                self.cuda_graph.capture_begin()
                with sdpa_kernel([SDPBackend.MATH]):
                    out = self.decode_one_token(
                        self.static_input,
                        None,
                        cache_position=self.cache_position + 1,
                        past_key_values=self.past_key_values,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                    self.static_output.copy_(out)
                self.cuda_graph.capture_end()
            torch.cuda.synchronize()
            
            #Turn-off
            self.do_capture_graph = False
        else:
            self.static_input.copy_(next_token)
            self.cuda_graph.replay()

        next_token = self.static_output
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def print_current_token(self, output_text_len):
        output_text = self.tokenizer.decode(self.generated_ids[0, self.begin_gen_position : self.cache_position + 1])
        printable_text = output_text[output_text_len:]
        output_text_len = len(output_text)
        print(printable_text, end="", flush=True)
        return output_text_len

    def next_token_iterator(self, next_token, max_new_tokens, verbose, print_tokens, cleanup=True):
        output_text, output_text_len = "", 0
        for i in tqdm(range(1, max_new_tokens), disable=(not verbose or print_tokens)):
            next_token = self.gen_next_token(next_token)

            if next_token[0].item() == self.tokenizer.eos_token_id:
                break

            # You need to keep track of the whole text, otherwise you lose spaces, makes everything much slower ¯\_(ツ)_/¯
            # https://github.com/huggingface/transformers/blob/b109257f4fb8b1166e7c53cc5418632014ed53a5/src/transformers/generation/streamers.py#L95-L114
            if print_tokens:
                output_text_len = self.print_current_token(output_text_len)

        torch.cuda.synchronize()
        input_tokens  = self.generated_ids[0, : self.begin_gen_position].cpu()
        output_tokens = self.generated_ids[0, self.begin_gen_position : self.cache_position].cpu()
        output_text   = self.tokenizer.decode(output_tokens)

        if cleanup:
            # model._reset_cache()
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return {
            "output_text": output_text,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
        }

    @torch.inference_mode()
    def generate(self, prompt, use_chat_template=True, verbose=True, print_tokens=False):
        self.setup(
            inputs=self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            max_new_tokens=self.max_new_tokens,
        )
        return self.next_token_iterator(self.prefill(), self.max_new_tokens, verbose, print_tokens)

    def generate_(self, prompt, use_chat_template=True, verbose=False, print_tokens=False):
        gen_out = self.model.generate(
            **self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            do_sample=self.do_sample,
            cache_implementation="static",
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_k,
            # use_cache=False,
        )[0]

        return {"output_text": self.tokenizer.decode(gen_out), "output_tokens": gen_out}

    def tokenize_prompt(self, prompt, use_chat_template=True):
        if use_chat_template:

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        return self.tokenizer([prompt], return_tensors="pt").to(device=self.model.device)
